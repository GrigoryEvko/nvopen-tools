// Function: sub_1A174E0
// Address: 0x1a174e0
//
__int64 __fastcall sub_1A174E0(
        __int64 a1,
        double a2,
        double a3,
        double a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  unsigned int v9; // edx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r12
  _QWORD *v13; // r14
  _QWORD *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // r13
  __int64 v20; // r15
  _QWORD *v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 result; // rax
  _QWORD *v25; // r13
  __int64 v26; // r12
  _QWORD *v27; // r14
  _QWORD *v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rdx
  _QWORD *v32; // rax
  __int64 v33; // r13
  __int64 v34; // r15
  _QWORD *v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // r13
  __int64 v43; // rsi

  v9 = *(_DWORD *)(a1 + 824);
  if ( !*(_DWORD *)(a1 + 1880) )
  {
    result = *(unsigned int *)(a1 + 1352);
    goto LABEL_63;
  }
  if ( !v9 )
    goto LABEL_27;
  do
  {
    while ( 1 )
    {
      do
      {
        v10 = v9--;
        v11 = *(_QWORD *)(*(_QWORD *)(a1 + 816) + 8 * v10 - 8);
        *(_DWORD *)(a1 + 824) = v9;
        v12 = *(_QWORD *)(v11 + 8);
        if ( v12 )
        {
          while ( 1 )
          {
            v18 = sub_1648700(v12);
            v19 = (__int64)v18;
            if ( *((_BYTE *)v18 + 16) > 0x17u )
              break;
LABEL_10:
            v12 = *(_QWORD *)(v12 + 8);
            if ( !v12 )
            {
              v9 = *(_DWORD *)(a1 + 824);
              goto LABEL_26;
            }
          }
          v20 = v18[5];
          v21 = *(_QWORD **)(a1 + 32);
          v14 = *(_QWORD **)(a1 + 24);
          if ( v21 == v14 )
          {
            v13 = &v14[*(unsigned int *)(a1 + 44)];
            if ( v14 == v13 )
            {
              v17 = *(_QWORD *)(a1 + 24);
            }
            else
            {
              do
              {
                if ( v20 == *v14 )
                  break;
                ++v14;
              }
              while ( v13 != v14 );
              v17 = (__int64)v13;
            }
          }
          else
          {
            v13 = &v21[*(unsigned int *)(a1 + 40)];
            v14 = sub_16CC9F0(a1 + 16, v20);
            if ( v20 == *v14 )
            {
              v22 = *(_QWORD *)(a1 + 32);
              if ( v22 == *(_QWORD *)(a1 + 24) )
                v23 = *(unsigned int *)(a1 + 44);
              else
                v23 = *(unsigned int *)(a1 + 40);
              v17 = v22 + 8 * v23;
            }
            else
            {
              v16 = *(_QWORD *)(a1 + 32);
              if ( v16 != *(_QWORD *)(a1 + 24) )
              {
                v17 = *(unsigned int *)(a1 + 40);
                v14 = (_QWORD *)(v16 + 8 * v17);
                goto LABEL_8;
              }
              v14 = (_QWORD *)(v16 + 8LL * *(unsigned int *)(a1 + 44));
              v17 = (__int64)v14;
            }
          }
          while ( (_QWORD *)v17 != v14 && *v14 >= 0xFFFFFFFFFFFFFFFELL )
            ++v14;
LABEL_8:
          if ( v14 != v13 )
            sub_1A15E70(a1, v19, a2, a3, a4, v17, v15, a8, a9);
          goto LABEL_10;
        }
LABEL_26:
        ;
      }
      while ( v9 );
LABEL_27:
      result = *(unsigned int *)(a1 + 1352);
      if ( (_DWORD)result )
        goto LABEL_30;
LABEL_56:
      LODWORD(v38) = *(_DWORD *)(a1 + 1880);
      if ( (_DWORD)v38 )
      {
        do
        {
          v39 = (unsigned int)v38;
          v38 = (unsigned int)(v38 - 1);
          v40 = *(_QWORD *)(*(_QWORD *)(a1 + 1872) + 8 * v39 - 8);
          *(_DWORD *)(a1 + 1880) = v38;
          v41 = *(_QWORD *)(v40 + 48);
          v42 = v40 + 40;
          if ( v40 + 40 != v41 )
          {
            do
            {
              v43 = v41;
              v41 = *(_QWORD *)(v41 + 8);
              sub_1A15E70(a1, v43 - 24, a2, a3, a4, v38, v39, a8, a9);
            }
            while ( v42 != v41 );
            LODWORD(v38) = *(_DWORD *)(a1 + 1880);
          }
        }
        while ( (_DWORD)v38 );
        result = *(unsigned int *)(a1 + 1352);
      }
      v9 = *(_DWORD *)(a1 + 824);
LABEL_63:
      if ( !(_DWORD)result )
        break;
      if ( !v9 )
      {
LABEL_30:
        while ( 1 )
        {
          v25 = *(_QWORD **)(*(_QWORD *)(a1 + 1344) + 8LL * (unsigned int)result - 8);
          *(_DWORD *)(a1 + 1352) = result - 1;
          if ( *(_BYTE *)(*v25 + 8LL) == 13 || ((*(_BYTE *)sub_1A10F60(a1, (__int64)v25) ^ 6) & 6) != 0 )
          {
            v26 = v25[1];
            if ( v26 )
              break;
          }
LABEL_29:
          result = *(unsigned int *)(a1 + 1352);
          if ( !(_DWORD)result )
            goto LABEL_56;
        }
        while ( 1 )
        {
          v32 = sub_1648700(v26);
          v33 = (__int64)v32;
          if ( *((_BYTE *)v32 + 16) > 0x17u )
            break;
LABEL_39:
          v26 = *(_QWORD *)(v26 + 8);
          if ( !v26 )
            goto LABEL_29;
        }
        v34 = v32[5];
        v35 = *(_QWORD **)(a1 + 32);
        v28 = *(_QWORD **)(a1 + 24);
        if ( v35 == v28 )
        {
          v27 = &v28[*(unsigned int *)(a1 + 44)];
          if ( v28 == v27 )
          {
            v31 = *(_QWORD *)(a1 + 24);
          }
          else
          {
            do
            {
              if ( v34 == *v28 )
                break;
              ++v28;
            }
            while ( v27 != v28 );
            v31 = (__int64)v27;
          }
        }
        else
        {
          v27 = &v35[*(unsigned int *)(a1 + 40)];
          v28 = sub_16CC9F0(a1 + 16, v34);
          if ( v34 == *v28 )
          {
            v36 = *(_QWORD *)(a1 + 32);
            if ( v36 == *(_QWORD *)(a1 + 24) )
              v37 = *(unsigned int *)(a1 + 44);
            else
              v37 = *(unsigned int *)(a1 + 40);
            v31 = v36 + 8 * v37;
          }
          else
          {
            v30 = *(_QWORD *)(a1 + 32);
            if ( v30 != *(_QWORD *)(a1 + 24) )
            {
              v31 = *(unsigned int *)(a1 + 40);
              v28 = (_QWORD *)(v30 + 8 * v31);
              goto LABEL_37;
            }
            v28 = (_QWORD *)(v30 + 8LL * *(unsigned int *)(a1 + 44));
            v31 = (__int64)v28;
          }
        }
        while ( (_QWORD *)v31 != v28 && *v28 >= 0xFFFFFFFFFFFFFFFELL )
          ++v28;
LABEL_37:
        if ( v28 != v27 )
          sub_1A15E70(a1, v33, a2, a3, a4, v31, v29, a8, a9);
        goto LABEL_39;
      }
    }
  }
  while ( v9 );
  return result;
}
