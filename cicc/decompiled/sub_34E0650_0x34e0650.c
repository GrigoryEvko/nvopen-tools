// Function: sub_34E0650
// Address: 0x34e0650
//
__int64 __fastcall sub_34E0650(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r10
  __int64 v4; // r15
  __int64 (*v5)(void); // rax
  __int64 result; // rax
  __int64 v7; // r15
  unsigned __int8 *v8; // r12
  unsigned int v9; // r13d
  __int64 v10; // r12
  int v11; // r14d
  __int64 v12; // r13
  __int64 v13; // r10
  __int64 v14; // rsi
  __int64 *v15; // rdx
  __int64 v16; // rdx
  char *v17; // rsi
  __int64 v18; // rdx
  _DWORD *v19; // rcx
  __int64 v20; // rdx
  unsigned int v21; // r13d
  int v22; // r14d
  __int16 *v23; // rsi
  __int64 v24; // r9
  int v25; // eax
  unsigned int v26; // eax
  int v27; // edx
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // rax
  unsigned __int16 v32; // r15
  __int16 *v33; // r13
  int v34; // eax
  __int64 v35; // rax
  __int16 *v36; // rax
  __int16 *v37; // rcx
  int v38; // esi
  __int64 v39; // rax
  __int16 *v40; // rdx
  __int64 v41; // rax
  __int64 v42; // [rsp+0h] [rbp-80h]
  unsigned int v43; // [rsp+Ch] [rbp-74h]
  __int64 v44; // [rsp+10h] [rbp-70h]
  __int64 v45; // [rsp+20h] [rbp-60h]
  __int64 v46; // [rsp+28h] [rbp-58h]
  __int64 v47; // [rsp+30h] [rbp-50h]
  __int64 v48; // [rsp+30h] [rbp-50h]
  unsigned int v49; // [rsp+38h] [rbp-48h]
  unsigned int v51; // [rsp+44h] [rbp-3Ch] BYREF
  unsigned int v52[14]; // [rsp+48h] [rbp-38h] BYREF

  v3 = a2;
  v4 = a1;
  v5 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 24) + 920LL);
  if ( v5 != sub_2DB1B30 )
  {
    result = v5();
    v3 = a2;
    if ( (_BYTE)result )
    {
LABEL_13:
      v10 = 0;
      v11 = *(_DWORD *)(v3 + 40) & 0xFFFFFF;
      v12 = v3;
      if ( !v11 )
        return result;
      while ( 1 )
      {
        result = 5 * v10;
        v13 = *(_QWORD *)(v12 + 32) + 40 * v10;
        if ( *(_BYTE *)v13 )
          goto LABEL_15;
        result = *(unsigned int *)(v13 + 8);
        v51 = result;
        if ( !(_DWORD)result || (*(_BYTE *)(v13 + 3) & 0x10) != 0 )
          goto LABEL_15;
        v14 = *(_QWORD *)(v12 + 16);
        if ( *(unsigned __int16 *)(v14 + 2) > (unsigned int)v10 )
        {
          v48 = v13;
          v41 = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(v4 + 24) + 16LL))(
                  *(_QWORD *)(v4 + 24),
                  v14,
                  (unsigned int)v10,
                  *(_QWORD *)(v4 + 32),
                  *(_QWORD *)(v4 + 8));
          v13 = v48;
          v15 = (__int64 *)(*(_QWORD *)(v4 + 120) + 8LL * v51);
          if ( *v15 )
          {
            if ( v41 && *v15 == v41 )
              goto LABEL_22;
          }
          else if ( v41 )
          {
            *v15 = v41;
            goto LABEL_22;
          }
        }
        else
        {
          v15 = (__int64 *)(*(_QWORD *)(v4 + 120) + 8 * result);
        }
        *v15 = -1;
LABEL_22:
        *(_QWORD *)v52 = v13;
        sub_34E0050((_QWORD *)(v4 + 144), &v51, v52);
        result = (__int64)sub_E922F0(*(_QWORD **)(v4 + 32), v51);
        v17 = (char *)(result + 2 * v16);
        if ( (char *)result == v17 )
        {
LABEL_15:
          if ( v11 == (_DWORD)++v10 )
            return result;
        }
        else
        {
          do
          {
            v18 = *(unsigned __int16 *)result;
            v19 = (_DWORD *)(*(_QWORD *)(v4 + 192) + 4 * v18);
            if ( *v19 == -1 )
            {
              *v19 = a3;
              *(_DWORD *)(*(_QWORD *)(v4 + 216) + 4 * v18) = -1;
            }
            result += 2;
          }
          while ( (char *)result != v17 );
          if ( v11 == (_DWORD)++v10 )
            return result;
        }
      }
    }
  }
  result = *(_DWORD *)(v3 + 40) & 0xFFFFFF;
  if ( (*(_DWORD *)(v3 + 40) & 0xFFFFFF) != 0 )
  {
    v47 = v3;
    v7 = 0;
    v46 = 40LL * (unsigned int)result;
    while ( 1 )
    {
      v8 = (unsigned __int8 *)(v7 + *(_QWORD *)(v47 + 32));
      result = *v8;
      if ( (_BYTE)result != 12 )
        goto LABEL_5;
      v20 = *(_QWORD *)(a1 + 32);
      if ( *(_DWORD *)(v20 + 16) != 1 )
        break;
LABEL_11:
      v7 += 40;
      if ( v46 == v7 )
      {
        v3 = v47;
        v4 = a1;
        goto LABEL_13;
      }
    }
    v21 = 1;
    v22 = *(_DWORD *)(v20 + 16);
    while ( 1 )
    {
      v23 = (__int16 *)(*(_QWORD *)(v20 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v20 + 8) + 24LL * v21 + 4));
      if ( !v23 )
        goto LABEL_30;
      v24 = *((_QWORD *)v8 + 3);
      v25 = *(_DWORD *)(v24 + 4LL * ((unsigned __int16)v21 >> 5));
      if ( !_bittest(&v25, v21) )
        break;
LABEL_31:
      if ( ++v21 == v22 )
      {
LABEL_39:
        result = *v8;
LABEL_5:
        if ( !(_BYTE)result )
        {
          v9 = *((_DWORD *)v8 + 2);
          if ( v9 )
          {
            if ( (v8[3] & 0x10) != 0 )
            {
              result = v7 + *(_QWORD *)(v47 + 32);
              if ( *(_BYTE *)result || (*(_BYTE *)(result + 3) & 0x10) == 0 || (*(_WORD *)(result + 2) & 0xFF0) == 0 )
              {
                v28 = *(_QWORD *)(a1 + 32);
                v45 = *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL * (v9 >> 6)) & (1LL << v9);
                v44 = 24LL * v9;
                v29 = *(_QWORD *)(v28 + 56);
                v30 = *(unsigned int *)(*(_QWORD *)(v28 + 8) + v44 + 4);
                v31 = *(_QWORD *)(v28 + 8) + v44;
                if ( v29 + 2 * v30 )
                {
                  v49 = *((_DWORD *)v8 + 2);
                  v43 = v49;
                  v42 = v7;
                  v32 = v49;
                  v33 = (__int16 *)(v29 + 2 * v30);
                  while ( 1 )
                  {
                    *(_DWORD *)(*(_QWORD *)(a1 + 216) + 4LL * v32) = a3;
                    *(_DWORD *)(*(_QWORD *)(a1 + 192) + 4LL * v32) = -1;
                    *(_QWORD *)(*(_QWORD *)(a1 + 120) + 8LL * v32) = 0;
                    v52[0] = v32;
                    sub_34E0580((_QWORD *)(a1 + 144), v52);
                    if ( !v45 )
                      *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL * (v32 >> 6)) &= ~(1LL << v32);
                    v34 = *v33++;
                    if ( !(_WORD)v34 )
                      break;
                    v49 += v34;
                    v32 = v49;
                  }
                  v35 = *(_QWORD *)(a1 + 32);
                  v9 = v43;
                  v7 = v42;
                  v29 = *(_QWORD *)(v35 + 56);
                  v31 = *(_QWORD *)(v35 + 8) + v44;
                }
                v36 = (__int16 *)(v29 + 2LL * *(unsigned int *)(v31 + 8));
                v37 = v36 + 1;
                result = (unsigned int)*v36;
                v38 = v9 + result;
                if ( (_WORD)result )
                {
                  v39 = (unsigned __int16)v38;
                  v40 = v37;
                  while ( 1 )
                  {
                    ++v40;
                    *(_QWORD *)(*(_QWORD *)(a1 + 120) + 8 * v39) = -1;
                    result = (unsigned int)*(v40 - 1);
                    if ( !*(v40 - 1) )
                      break;
                    v38 += result;
                    v39 = (unsigned __int16)v38;
                  }
                }
              }
            }
          }
        }
        goto LABEL_11;
      }
LABEL_32:
      v20 = *(_QWORD *)(a1 + 32);
    }
    v26 = v21;
    while ( 1 )
    {
      v26 += *v23;
      if ( !*v23 )
        break;
      ++v23;
      v27 = *(_DWORD *)(v24 + 4LL * ((unsigned __int16)v26 >> 5));
      if ( _bittest(&v27, v26) )
      {
        if ( ++v21 != v22 )
          goto LABEL_32;
        goto LABEL_39;
      }
    }
LABEL_30:
    *(_DWORD *)(*(_QWORD *)(a1 + 216) + 4LL * v21) = a3;
    *(_DWORD *)(*(_QWORD *)(a1 + 192) + 4LL * v21) = -1;
    *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL * (v21 >> 6)) &= ~(1LL << v21);
    *(_QWORD *)(*(_QWORD *)(a1 + 120) + 8LL * v21) = 0;
    v52[0] = v21;
    sub_34E0580((_QWORD *)(a1 + 144), v52);
    goto LABEL_31;
  }
  return result;
}
