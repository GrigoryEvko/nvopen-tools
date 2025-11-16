// Function: sub_1E7B680
// Address: 0x1e7b680
//
__int64 __fastcall sub_1E7B680(_QWORD *a1, int a2, __int64 a3, __int64 a4, unsigned __int64 a5, _QWORD *a6)
{
  bool v11; // si
  __int64 result; // rax
  __int64 v13; // rax
  int v14; // edi
  __int64 v15; // r8
  int v16; // edi
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // r10
  _QWORD **v20; // rax
  _QWORD *v21; // rax
  unsigned int i; // edx
  unsigned int v23; // ecx
  __int64 *v24; // rax
  __int64 v25; // r10
  _QWORD **v26; // rax
  _QWORD *v27; // rax
  unsigned int j; // ecx
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rcx
  bool v32; // di
  __int64 v33; // rdx
  __int64 v34; // r8
  int v35; // eax
  int v36; // r9d
  int v37; // eax
  int v38; // r9d
  char v40; // [rsp-39h] [rbp-39h] BYREF

  if ( a4 == a5 )
    return 0;
  v11 = sub_1E5EBB0(*(_QWORD *)(a1[33] + 232LL), a5, a4);
  if ( v11 )
  {
    v13 = a1[34];
    v14 = *(_DWORD *)(v13 + 256);
    if ( !v14 )
      goto LABEL_15;
    v15 = *(_QWORD *)(v13 + 240);
    v16 = v14 - 1;
    v17 = v16 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
    v18 = (__int64 *)(v15 + 16LL * v17);
    v19 = *v18;
    if ( a4 == *v18 )
    {
LABEL_7:
      v20 = (_QWORD **)v18[1];
      if ( v20 )
      {
        v21 = *v20;
        for ( i = 1; v21; ++i )
          v21 = (_QWORD *)*v21;
LABEL_10:
        v23 = v16 & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
        v24 = (__int64 *)(v15 + 16LL * v23);
        v25 = *v24;
        if ( a5 == *v24 )
        {
LABEL_11:
          v26 = (_QWORD **)v24[1];
          if ( v26 )
          {
            v27 = *v26;
            for ( j = 1; v27; ++j )
              v27 = (_QWORD *)*v27;
LABEL_14:
            if ( j < i )
              return 1;
LABEL_15:
            v29 = a1[31];
            if ( a2 < 0 )
              v30 = *(_QWORD *)(*(_QWORD *)(v29 + 24) + 16LL * (a2 & 0x7FFFFFFF) + 8);
            else
              v30 = *(_QWORD *)(*(_QWORD *)(v29 + 272) + 8LL * (unsigned int)a2);
            while ( v30 )
            {
              if ( (*(_BYTE *)(v30 + 3) & 0x10) == 0 && (*(_BYTE *)(v30 + 4) & 8) == 0 )
              {
                v31 = *(_QWORD *)(v30 + 16);
                v32 = 0;
                if ( a5 == *(_QWORD *)(v31 + 24) )
                  goto LABEL_28;
LABEL_23:
                while ( 1 )
                {
                  v30 = *(_QWORD *)(v30 + 32);
                  if ( !v30 )
                    break;
                  while ( (*(_BYTE *)(v30 + 3) & 0x10) == 0 )
                  {
                    if ( (*(_BYTE *)(v30 + 4) & 8) != 0 )
                      break;
                    v33 = *(_QWORD *)(v30 + 16);
                    if ( v31 == v33 )
                      break;
                    v31 = *(_QWORD *)(v30 + 16);
                    if ( a5 != *(_QWORD *)(v33 + 24) )
                      break;
LABEL_28:
                    if ( **(_WORD **)(v31 + 16) == 45 )
                      goto LABEL_23;
                    v30 = *(_QWORD *)(v30 + 32);
                    if ( **(_WORD **)(v31 + 16) )
                      v32 = v11;
                    if ( !v30 )
                      goto LABEL_32;
                  }
                }
LABEL_32:
                if ( v32 )
                {
                  v40 = 0;
                  v34 = sub_1E7AE00(a1, a3, a5, &v40, a6);
                  result = 0;
                  if ( v34 )
                    return sub_1E7B680(a1, (unsigned int)a2, a3, a5, v34, a6);
                  return result;
                }
                return 1;
              }
              v30 = *(_QWORD *)(v30 + 32);
            }
            return 1;
          }
        }
        else
        {
          v35 = 1;
          while ( v25 != -8 )
          {
            v36 = v35 + 1;
            v23 = v16 & (v35 + v23);
            v24 = (__int64 *)(v15 + 16LL * v23);
            v25 = *v24;
            if ( a5 == *v24 )
              goto LABEL_11;
            v35 = v36;
          }
        }
        j = 0;
        goto LABEL_14;
      }
    }
    else
    {
      v37 = 1;
      while ( v19 != -8 )
      {
        v38 = v37 + 1;
        v17 = v16 & (v37 + v17);
        v18 = (__int64 *)(v15 + 16LL * v17);
        v19 = *v18;
        if ( a4 == *v18 )
          goto LABEL_7;
        v37 = v38;
      }
    }
    i = 0;
    goto LABEL_10;
  }
  return 1;
}
