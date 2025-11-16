// Function: sub_20C7600
// Address: 0x20c7600
//
__int64 *__fastcall sub_20C7600(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rbx
  unsigned __int8 v9; // dl
  __int64 *v10; // r15
  __int64 (*v11)(); // rax
  unsigned __int8 v12; // al
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rsi
  int v15; // r9d
  int v16; // r8d
  _DWORD *v17; // rax
  __int64 v18; // r10
  _DWORD *v19; // rcx
  _DWORD *v20; // rdx
  unsigned __int64 v21; // r15
  _DWORD *v22; // rdx
  _DWORD *i; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // r15
  _QWORD *v27; // rdi
  __int64 v28; // rax
  int v29; // eax
  unsigned int v30; // eax
  unsigned __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rcx
  __int64 v34; // r10
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // [rsp+0h] [rbp-70h]
  __int64 v38; // [rsp+8h] [rbp-68h]
  __int64 v39; // [rsp+10h] [rbp-60h]
  int v40; // [rsp+10h] [rbp-60h]
  int v42; // [rsp+34h] [rbp-3Ch] BYREF
  _QWORD v43[7]; // [rsp+38h] [rbp-38h] BYREF

  v8 = a1;
  v9 = *(_BYTE *)(a1 + 16);
  if ( v9 > 0x17u )
  {
    while ( 1 )
    {
      if ( (*(_DWORD *)(v8 + 20) & 0xFFFFFFF) == 0 )
        return (__int64 *)v8;
      if ( (*(_BYTE *)(v8 + 23) & 0x40) != 0 )
      {
        v10 = **(__int64 ***)(v8 - 8);
        if ( v9 == 71 )
          goto LABEL_32;
      }
      else
      {
        v10 = *(__int64 **)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
        if ( v9 == 71 )
          goto LABEL_32;
      }
      if ( v9 == 56 )
        break;
      if ( v9 != 70 )
      {
        if ( v9 == 69 )
        {
          if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) == 16
            || 8 * (unsigned int)sub_15A9520(a5, 0) != *(_DWORD *)(*(_QWORD *)v8 + 8LL) >> 8 )
          {
            return (__int64 *)v8;
          }
        }
        else
        {
          if ( v9 == 60 )
          {
            v11 = *(__int64 (**)())(*(_QWORD *)a4 + 792LL);
            if ( v11 != sub_1F3CB90 )
            {
              if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v11)(a4, *v10, *(_QWORD *)v8) )
              {
                v30 = sub_1643030(*(_QWORD *)v8);
                if ( *a3 <= v30 )
                  v30 = *a3;
                *a3 = v30;
                goto LABEL_37;
              }
            }
          }
          v12 = *(_BYTE *)(v8 + 16);
          if ( v12 <= 0x17u )
            return (__int64 *)v8;
          switch ( v12 )
          {
            case 0x4Eu:
              v25 = v8 | 4;
              goto LABEL_45;
            case 0x1Du:
              v25 = v8 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_45:
              v26 = v25 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v25 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                return (__int64 *)v8;
              v27 = (_QWORD *)(v26 + 56);
              if ( (v25 & 4) != 0 )
              {
                if ( (unsigned __int8)sub_1560490(v27, 38, &v42) )
                {
                  v29 = v42;
                  if ( v42 )
                    goto LABEL_51;
                }
                v28 = *(_QWORD *)(v26 - 24);
                if ( *(_BYTE *)(v28 + 16) )
                  return (__int64 *)v8;
              }
              else
              {
                if ( (unsigned __int8)sub_1560490(v27, 38, &v42) )
                {
                  v29 = v42;
                  if ( v42 )
                  {
LABEL_51:
                    v10 = *(__int64 **)(v26
                                      + 24
                                      * ((unsigned int)(v29 - 1) - (unsigned __int64)(*(_DWORD *)(v26 + 20) & 0xFFFFFFF)));
                    if ( !v10 )
                      return (__int64 *)v8;
LABEL_32:
                    if ( !sub_20C7570(*v10, *(_QWORD *)v8, a4) )
                      return (__int64 *)v8;
                    goto LABEL_37;
                  }
                }
                v28 = *(_QWORD *)(v26 - 72);
                if ( *(_BYTE *)(v28 + 16) )
                  return (__int64 *)v8;
              }
              v43[0] = *(_QWORD *)(v28 + 112);
              if ( !(unsigned __int8)sub_1560490(v43, 38, &v42) )
                return (__int64 *)v8;
              v29 = v42;
              if ( !v42 )
                return (__int64 *)v8;
              goto LABEL_51;
            case 0x57u:
              v13 = *(unsigned int *)(v8 + 64);
              v14 = *(unsigned int *)(a2 + 8);
              v15 = *(_DWORD *)(v8 + 64);
              v16 = *(_DWORD *)(a2 + 8);
              if ( v13 <= v14 )
              {
                v17 = *(_DWORD **)(v8 + 56);
                v18 = *(_QWORD *)a2;
                v19 = &v17[v13];
                if ( v17 == v19 )
                {
LABEL_19:
                  v21 = v14 - v13;
                  if ( v14 > v14 - v13 )
                    goto LABEL_27;
                  if ( v14 < v13 )
                  {
                    if ( v21 > *(unsigned int *)(a2 + 12) )
                    {
                      v40 = *(_DWORD *)(a2 + 8);
                      sub_16CD150(a2, (const void *)(a2 + 16), v14 - v13, 4, v14, v13);
                      v16 = v40;
                      v15 = v13;
                      v18 = *(_QWORD *)a2;
                    }
                    v22 = (_DWORD *)(v18 + 4 * v21);
                    for ( i = (_DWORD *)(v18 + 4LL * *(unsigned int *)(a2 + 8)); v22 != i; ++i )
                    {
                      if ( i )
                        *i = 0;
                    }
LABEL_27:
                    *(_DWORD *)(a2 + 8) = v16 - v15;
                  }
                  v10 = *(__int64 **)(v8 - 24);
                  if ( !v10 )
                    return (__int64 *)v8;
                  goto LABEL_37;
                }
                v20 = (_DWORD *)(v18 + 4 * v14 - 4);
                while ( *v17 == *v20 )
                {
                  ++v17;
                  --v20;
                  if ( v19 == v17 )
                    goto LABEL_19;
                }
              }
              break;
            case 0x56u:
              v31 = *(unsigned int *)(v8 + 64);
              v32 = *(_QWORD *)(v8 + 56);
              v33 = *(unsigned int *)(a2 + 8);
              v34 = 4 * v31;
              v35 = v31;
              if ( v31 > (unsigned __int64)*(unsigned int *)(a2 + 12) - v33 )
              {
                v37 = *(unsigned int *)(v8 + 64);
                v38 = 4 * v31;
                v39 = *(_QWORD *)(v8 + 56);
                sub_16CD150(a2, (const void *)(a2 + 16), v31 + v33, 4, v31, v32);
                v34 = v38;
                v32 = v39;
                v31 = v37;
                v33 = *(unsigned int *)(a2 + 8);
                v35 = v37;
              }
              v36 = *(_QWORD *)a2 + 4 * v33;
              if ( v34 )
              {
                do
                {
                  v36 += 4;
                  *(_DWORD *)(v36 - 4) = *(_DWORD *)(v32 + v34 - 4 * v31 + 4 * v35-- - 4);
                }
                while ( v35 );
                LODWORD(v33) = *(_DWORD *)(a2 + 8);
              }
              *(_DWORD *)(a2 + 8) = v33 + v31;
              break;
            default:
              return (__int64 *)v8;
          }
        }
        goto LABEL_36;
      }
      if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) == 16 || 8 * (unsigned int)sub_15A9520(a5, 0) != *(_DWORD *)(*v10 + 8) >> 8 )
        return (__int64 *)v8;
LABEL_37:
      v9 = *((_BYTE *)v10 + 16);
      if ( v9 <= 0x17u )
        return v10;
      v8 = (__int64)v10;
    }
    if ( !(unsigned __int8)sub_15FA1F0(v8) )
      return (__int64 *)v8;
LABEL_36:
    if ( !v10 )
      return (__int64 *)v8;
    goto LABEL_37;
  }
  return (__int64 *)v8;
}
