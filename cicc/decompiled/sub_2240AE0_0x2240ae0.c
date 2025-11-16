// Function: sub_2240AE0
// Address: 0x2240ae0
//
void __fastcall sub_2240AE0(unsigned __int64 *a1, unsigned __int64 *a2)
{
  _QWORD *v2; // r14
  _BYTE *v4; // r13
  size_t v5; // r12
  unsigned __int64 v6; // rdx
  _BYTE *v7; // rsi
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rax
  unsigned __int64 v11[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( a1 != a2 )
  {
    v2 = a1 + 2;
    v4 = (_BYTE *)*a1;
    v5 = a2[1];
    if ( a1 + 2 == (unsigned __int64 *)*a1 )
      v6 = 15;
    else
      v6 = a1[2];
    if ( v6 < v5 )
    {
      v11[0] = a2[1];
      v8 = sub_22409D0((__int64)a1, v11, v6);
      v9 = *a1;
      v4 = (_BYTE *)v8;
      if ( v2 != (_QWORD *)*a1 )
        j___libc_free_0(v9);
      v10 = v11[0];
      *a1 = (unsigned __int64)v4;
      a1[2] = v10;
      if ( !v5 )
        goto LABEL_9;
    }
    else if ( !v5 )
    {
LABEL_9:
      a1[1] = v5;
      v4[v5] = 0;
      return;
    }
    v7 = (_BYTE *)*a2;
    if ( v5 == 1 )
      *v4 = *v7;
    else
      memcpy(v4, v7, v5);
    v4 = (_BYTE *)*a1;
    goto LABEL_9;
  }
}
