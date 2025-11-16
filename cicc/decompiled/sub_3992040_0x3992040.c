// Function: sub_3992040
// Address: 0x3992040
//
__int64 __fastcall sub_3992040(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rdi
  __int64 result; // rax
  __int64 v6; // rax
  unsigned int v7; // esi
  __int64 v8; // rdi
  __int64 v9; // r13
  int v10; // r8d
  __int64 v11; // rcx
  unsigned int v12; // r15d
  unsigned int v13; // edx
  __int64 *v14; // r9
  __int64 *v15; // r12
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rsi
  int v23; // edx
  int v24; // r10d
  int v25; // eax
  int v26; // eax
  int v27; // eax
  __int64 v28; // rsi
  int v29; // r8d
  __int64 *v30; // rdi
  unsigned int v31; // r15d
  __int64 v32; // rcx
  int v33; // r9d
  __int64 *v34; // r8

  v3 = *(_BYTE **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  result = (unsigned int)(unsigned __int8)*v3 - 17;
  if ( (unsigned __int8)(*v3 - 17) <= 2u )
  {
    v6 = sub_15B1030((__int64)v3);
    v7 = *(_DWORD *)(a1 + 664);
    v8 = a1 + 640;
    v9 = v6;
    if ( v7 )
    {
      v10 = v7 - 1;
      v11 = *(_QWORD *)(a1 + 648);
      v12 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
      v13 = (v7 - 1) & v12;
      LODWORD(v14) = 5 * v13;
      v15 = (__int64 *)(v11 + 88LL * v13);
      v16 = *v15;
      if ( v9 == *v15 )
      {
LABEL_4:
        v17 = *((unsigned int *)v15 + 4);
        if ( (unsigned int)v17 >= *((_DWORD *)v15 + 5) )
        {
          sub_16CD150((__int64)(v15 + 1), v15 + 3, 0, 8, v10, (int)v14);
          result = v15[1] + 8LL * *((unsigned int *)v15 + 4);
        }
        else
        {
          result = v15[1] + 8 * v17;
        }
LABEL_6:
        *(_QWORD *)result = a2;
        ++*((_DWORD *)v15 + 4);
        return result;
      }
      v24 = 1;
      v14 = 0;
      while ( v16 != -8 )
      {
        if ( v16 == -16 && !v14 )
          v14 = v15;
        v13 = v10 & (v24 + v13);
        v15 = (__int64 *)(v11 + 88LL * v13);
        v16 = *v15;
        if ( v9 == *v15 )
          goto LABEL_4;
        ++v24;
      }
      v25 = *(_DWORD *)(a1 + 656);
      if ( v14 )
        v15 = v14;
      ++*(_QWORD *)(a1 + 640);
      v23 = v25 + 1;
      if ( 4 * (v25 + 1) < 3 * v7 )
      {
        if ( v7 - *(_DWORD *)(a1 + 660) - v23 > v7 >> 3 )
        {
LABEL_12:
          *(_DWORD *)(a1 + 656) = v23;
          if ( *v15 != -8 )
            --*(_DWORD *)(a1 + 660);
          result = (__int64)(v15 + 3);
          *v15 = v9;
          v15[1] = (__int64)(v15 + 3);
          v15[2] = 0x800000000LL;
          goto LABEL_6;
        }
        sub_3991D30(v8, v7);
        v26 = *(_DWORD *)(a1 + 664);
        if ( v26 )
        {
          v27 = v26 - 1;
          v28 = *(_QWORD *)(a1 + 648);
          v29 = 1;
          v30 = 0;
          v31 = v27 & v12;
          v15 = (__int64 *)(v28 + 88LL * v31);
          v32 = *v15;
          v23 = *(_DWORD *)(a1 + 656) + 1;
          if ( v9 != *v15 )
          {
            while ( v32 != -8 )
            {
              if ( !v30 && v32 == -16 )
                v30 = v15;
              v31 = v27 & (v29 + v31);
              v15 = (__int64 *)(v28 + 88LL * v31);
              v32 = *v15;
              if ( v9 == *v15 )
                goto LABEL_12;
              ++v29;
            }
            if ( v30 )
              v15 = v30;
          }
          goto LABEL_12;
        }
LABEL_47:
        ++*(_DWORD *)(a1 + 656);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 640);
    }
    sub_3991D30(v8, 2 * v7);
    v18 = *(_DWORD *)(a1 + 664);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 648);
      v21 = (v18 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v15 = (__int64 *)(v20 + 88LL * v21);
      v22 = *v15;
      v23 = *(_DWORD *)(a1 + 656) + 1;
      if ( v9 != *v15 )
      {
        v33 = 1;
        v34 = 0;
        while ( v22 != -8 )
        {
          if ( !v34 && v22 == -16 )
            v34 = v15;
          v21 = v19 & (v33 + v21);
          v15 = (__int64 *)(v20 + 88LL * v21);
          v22 = *v15;
          if ( v9 == *v15 )
            goto LABEL_12;
          ++v33;
        }
        if ( v34 )
          v15 = v34;
      }
      goto LABEL_12;
    }
    goto LABEL_47;
  }
  return result;
}
