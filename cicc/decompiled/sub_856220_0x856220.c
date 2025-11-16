// Function: sub_856220
// Address: 0x856220
//
unsigned __int64 __fastcall sub_856220(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int16 v6; // ax
  int v7; // ebx
  __int64 v8; // r15
  _BOOL4 v9; // r12d
  __int64 v10; // rdi
  size_t v11; // rax
  __int64 v12; // r11
  unsigned int *v13; // rsi
  void *v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int64 result; // rax
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v23; // [rsp+8h] [rbp-48h]
  bool v24; // [rsp+17h] [rbp-39h]

  v6 = word_4F06418[0];
  if ( (unsigned __int16)(word_4F06418[0] - 9) <= 1u || (v24 = a1 != 0, word_4F06418[0] == 28) && a1 )
  {
    result = 1;
    v8 = 0;
  }
  else
  {
    v7 = 0;
    v8 = 0;
    v9 = 0;
    do
    {
      if ( v6 == 27 )
      {
        ++v7;
      }
      else
      {
        a5 = (unsigned int)(v7 == 0) + v7 - 1;
        if ( v6 == 28 )
          v7 = (v7 == 0) + v7 - 1;
      }
      v10 = qword_4F5FC08;
      v11 = v9 + v8 + qword_4F06400 + 1;
      v12 = qword_4F5FC08;
      if ( v11 > qword_4D03CC8 )
      {
        v21 = qword_4D03CC8 + 300;
        if ( v11 >= qword_4D03CC8 + 300 )
          v21 = v9 + v8 + qword_4F06400 + 1;
        v23 = v21;
        v12 = sub_822C60((void *)qword_4F5FC08, qword_4D03CC8, v21, (__int64)&qword_4D03CC8, a5, a6);
        qword_4F5FC08 = v12;
        v10 = v12;
        qword_4D03CC8 = v23;
      }
      if ( v9 )
        *(_BYTE *)(v12 + v8++) = 32;
      v13 = (unsigned int *)qword_4F06410;
      v14 = (void *)(v8 + v10);
      memcpy(v14, qword_4F06410, qword_4F06400);
      v8 += qword_4F06400;
      sub_7BC390();
      v9 = dword_4F063EC != 0;
      sub_7B8B50((unsigned __int64)v14, v13, v15, v16, v17, v18);
      v6 = word_4F06418[0];
    }
    while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u && (word_4F06418[0] != 28 || !v24 || v7) );
    result = v8 + 1;
  }
  v20 = qword_4F5FC08;
  if ( qword_4D03CC8 < result )
  {
    v22 = qword_4D03CC8 + 300;
    if ( qword_4D03CC8 + 300 < result )
      v22 = result;
    qword_4F5FC08 = sub_822C60((void *)qword_4F5FC08, qword_4D03CC8, v22, (__int64)&qword_4D03CC8, a5, a6);
    v20 = qword_4F5FC08;
    result = (unsigned __int64)&qword_4D03CC8;
    qword_4D03CC8 = v22;
  }
  *(_BYTE *)(v20 + v8) = 0;
  return result;
}
