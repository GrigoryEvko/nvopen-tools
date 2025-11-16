// Function: sub_27793B0
// Address: 0x27793b0
//
void __fastcall sub_27793B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rdi
  __int64 v8; // rcx
  int v9; // eax
  int v10; // esi
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r8
  __int64 v14; // rsi
  int v15; // eax

  if ( *(_QWORD *)(a1 + 104) )
  {
    if ( byte_4F8F8E8[0] )
      nullsub_390();
    v7 = *(_QWORD **)(a1 + 112);
    v8 = *(_QWORD *)(*v7 + 40LL);
    v9 = *(_DWORD *)(*v7 + 56LL);
    if ( v9 )
    {
      v10 = v9 - 1;
      v11 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v12 = (__int64 *)(v8 + 16LL * v11);
      v13 = *v12;
      if ( a2 == *v12 )
      {
LABEL_6:
        v14 = v12[1];
        if ( v14 )
          sub_D6E4B0(v7, v14, 1, v8, v13, a6);
      }
      else
      {
        v15 = 1;
        while ( v13 != -4096 )
        {
          a6 = (unsigned int)(v15 + 1);
          v11 = v10 & (v15 + v11);
          v12 = (__int64 *)(v8 + 16LL * v11);
          v13 = *v12;
          if ( a2 == *v12 )
            goto LABEL_6;
          v15 = a6;
        }
      }
    }
  }
}
