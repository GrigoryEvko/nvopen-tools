// Function: sub_16302F0
// Address: 0x16302f0
//
void __fastcall sub_16302F0(
        __int64 ***a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // r8
  unsigned int v12; // esi
  __int64 ****v13; // rdx
  __int64 ***v14; // r10
  __int64 ***v15; // r12
  int v16; // edx
  int v17; // r11d

  v9 = ***a1;
  v10 = *(unsigned int *)(v9 + 424);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD *)(v9 + 408);
    v12 = (v10 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v13 = (__int64 ****)(v11 + 16LL * v12);
    v14 = *v13;
    if ( a1 == *v13 )
    {
LABEL_3:
      if ( v13 != (__int64 ****)(v11 + 16 * v10) )
      {
        *v13 = (__int64 ***)-16LL;
        v15 = v13[1];
        --*(_DWORD *)(v9 + 416);
        ++*(_DWORD *)(v9 + 420);
        sub_16302D0((const __m128i *)(v15 + 1), 0, a2, a3, a4, a5, a6, a7, a8, a9);
        if ( v15 )
        {
          if ( ((_BYTE)v15[4] & 1) == 0 )
            j___libc_free_0(v15[5]);
          j_j___libc_free_0(v15, 144);
        }
      }
    }
    else
    {
      v16 = 1;
      while ( v14 != (__int64 ***)-8LL )
      {
        v17 = v16 + 1;
        v12 = (v10 - 1) & (v16 + v12);
        v13 = (__int64 ****)(v11 + 16LL * v12);
        v14 = *v13;
        if ( a1 == *v13 )
          goto LABEL_3;
        v16 = v17;
      }
    }
  }
}
