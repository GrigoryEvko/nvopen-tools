// Function: sub_278C2C0
// Address: 0x278c2c0
//
__int64 __fastcall sub_278C2C0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  _QWORD *v9; // rdi
  __int64 v10; // rcx
  int v11; // eax
  int v12; // esi
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r8
  __int64 v16; // rsi
  int v18; // eax

  v8 = a1[2];
  if ( v8 )
    sub_1031600(v8, (__int64)a2);
  v9 = (_QWORD *)a1[15];
  if ( v9 )
  {
    v10 = *(_QWORD *)(*v9 + 40LL);
    v11 = *(_DWORD *)(*v9 + 56LL);
    if ( v11 )
    {
      v12 = v11 - 1;
      v13 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = (__int64 *)(v10 + 16LL * v13);
      v15 = *v14;
      if ( a2 == (_QWORD *)*v14 )
      {
LABEL_6:
        v16 = v14[1];
        if ( v16 )
          sub_D6E4B0(v9, v16, 0, v10, v15, a6);
      }
      else
      {
        v18 = 1;
        while ( v15 != -4096 )
        {
          a6 = (unsigned int)(v18 + 1);
          v13 = v12 & (v18 + v13);
          v14 = (__int64 *)(v10 + 16LL * v13);
          v15 = *v14;
          if ( a2 == (_QWORD *)*v14 )
            goto LABEL_6;
          v18 = a6;
        }
      }
    }
  }
  sub_30EC400(a1[13], a2);
  return sub_B43D60(a2);
}
