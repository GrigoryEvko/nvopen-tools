// Function: sub_BA0700
// Address: 0xba0700
//
__int64 __fastcall sub_BA0700(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  _BYTE *v4; // rax
  int v5; // r15d
  __int64 v6; // rax
  __int64 *v7; // rdx
  unsigned int v8; // eax
  __int64 v9; // rcx
  int v10; // r12d
  int v11; // r9d
  int v12; // r14d
  unsigned int i; // r12d
  __int64 *v14; // r15
  __int64 v15; // r13
  _BYTE *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  unsigned int v20; // edi
  __int64 *v21; // rsi
  __int64 v22; // rdx
  unsigned int v23; // edi
  __int64 *v24; // rsi
  __int64 v25; // rax
  unsigned int v26; // r12d
  _BYTE *v27; // rax
  __int64 v28; // rdx
  _BYTE *v29; // rax
  char v30; // al
  _BYTE *v31; // rax
  char v32; // al
  __int64 result; // rax
  __int64 v34; // rcx
  __int64 v35; // rax
  unsigned int v36; // edi
  __int64 *v37; // rsi
  __int64 v38; // rdx
  unsigned int v39; // edi
  __int64 *v40; // rsi
  __int64 v41; // rax
  unsigned int v42; // esi
  int v43; // eax
  _QWORD *v44; // rdx
  int v45; // eax
  int v46; // [rsp+0h] [rbp-80h]
  __int64 v47; // [rsp+10h] [rbp-70h]
  __int64 v48[2]; // [rsp+18h] [rbp-68h] BYREF
  _QWORD *v49; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v50; // [rsp+30h] [rbp-50h] BYREF
  __int64 v51; // [rsp+38h] [rbp-48h] BYREF
  __int64 v52; // [rsp+40h] [rbp-40h] BYREF
  __int64 v53[7]; // [rsp+48h] [rbp-38h] BYREF

  v48[0] = a1;
  v3 = *(_QWORD *)sub_A17150((_BYTE *)(a1 - 16));
  v50 = (_QWORD *)v3;
  v51 = *((_QWORD *)sub_A17150((_BYTE *)(a1 - 16)) + 1);
  v52 = *((_QWORD *)sub_A17150((_BYTE *)(a1 - 16)) + 2);
  v4 = sub_A17150((_BYTE *)(a1 - 16));
  v5 = *(_DWORD *)(a2 + 24);
  v53[0] = *((_QWORD *)v4 + 3);
  v47 = *(_QWORD *)(a2 + 8);
  if ( v5 )
  {
    if ( v3 && *(_BYTE *)v3 == 1 )
    {
      v6 = *(_QWORD *)(v3 + 136);
      v7 = *(__int64 **)(v6 + 24);
      v8 = *(_DWORD *)(v6 + 32);
      if ( v8 > 0x40 )
      {
        v9 = *v7;
      }
      else
      {
        v9 = 0;
        if ( v8 )
          v9 = (__int64)((_QWORD)v7 << (64 - (unsigned __int8)v8)) >> (64 - (unsigned __int8)v8);
      }
      v49 = (_QWORD *)v9;
      v10 = sub_AF7D50((__int64 *)&v49, &v51, &v52, v53);
    }
    else
    {
      v10 = sub_AF81D0((__int64 *)&v50, &v51, &v52, v53);
    }
    v11 = v5 - 1;
    v12 = 1;
    for ( i = (v5 - 1) & v10; ; i = v11 & v26 )
    {
      v14 = (__int64 *)(v47 + 8LL * i);
      v15 = *v14;
      if ( *v14 == -4096 )
        break;
      if ( v15 != -8192 )
      {
        v46 = v11;
        v16 = sub_A17150((_BYTE *)(v15 - 16));
        v11 = v46;
        v17 = *(_QWORD *)v16;
        if ( *(_QWORD **)v16 == v50 )
          goto LABEL_24;
        if ( v50 && *(_BYTE *)v50 == 1 && v17 && *(_BYTE *)v17 == 1 )
        {
          v18 = v50[17];
          v19 = *(_QWORD *)(v17 + 136);
          v20 = *(_DWORD *)(v18 + 32);
          v21 = *(__int64 **)(v18 + 24);
          if ( v20 > 0x40 )
          {
            v22 = *v21;
          }
          else
          {
            v22 = 0;
            if ( v20 )
              v22 = (__int64)((_QWORD)v21 << (64 - (unsigned __int8)v20)) >> (64 - (unsigned __int8)v20);
          }
          v23 = *(_DWORD *)(v19 + 32);
          v24 = *(__int64 **)(v19 + 24);
          if ( v23 > 0x40 )
          {
            v25 = *v24;
          }
          else
          {
            v25 = 0;
            if ( v23 )
              v25 = (__int64)((_QWORD)v24 << (64 - (unsigned __int8)v23)) >> (64 - (unsigned __int8)v23);
          }
          if ( v25 == v22 )
          {
LABEL_24:
            v27 = sub_A17150((_BYTE *)(v15 - 16));
            v11 = v46;
            v28 = *((_QWORD *)v27 + 1);
            if ( v28 == v51 )
              goto LABEL_59;
            if ( v51 && *(_BYTE *)v51 == 1 && v28 && *(_BYTE *)v28 == 1 )
            {
              v34 = *(_QWORD *)(v51 + 136);
              v35 = *(_QWORD *)(v28 + 136);
              v36 = *(_DWORD *)(v34 + 32);
              v37 = *(__int64 **)(v34 + 24);
              if ( v36 <= 0x40 )
              {
                v38 = 0;
                if ( v36 )
                  v38 = (__int64)((_QWORD)v37 << (64 - (unsigned __int8)v36)) >> (64 - (unsigned __int8)v36);
              }
              else
              {
                v38 = *v37;
              }
              v39 = *(_DWORD *)(v35 + 32);
              v40 = *(__int64 **)(v35 + 24);
              if ( v39 > 0x40 )
              {
                v41 = *v40;
              }
              else
              {
                v41 = 0;
                if ( v39 )
                  v41 = (__int64)((_QWORD)v40 << (64 - (unsigned __int8)v39)) >> (64 - (unsigned __int8)v39);
              }
              if ( v41 == v38 )
              {
LABEL_59:
                v29 = sub_A17150((_BYTE *)(v15 - 16));
                v30 = sub_B8F9E0(v52, *((_QWORD *)v29 + 2));
                v11 = v46;
                if ( v30 )
                {
                  v31 = sub_A17150((_BYTE *)(v15 - 16));
                  v32 = sub_B8F9E0(v53[0], *((_QWORD *)v31 + 3));
                  v11 = v46;
                  if ( v32 )
                  {
                    if ( v14 != (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) )
                    {
                      result = v15;
                      if ( v15 )
                        return result;
                    }
                    break;
                  }
                }
              }
            }
          }
        }
      }
      v26 = v12 + i;
      ++v12;
    }
  }
  if ( !(unsigned __int8)sub_AFC4C0(a2, v48, &v49) )
  {
    v42 = *(_DWORD *)(a2 + 24);
    v43 = *(_DWORD *)(a2 + 16);
    v44 = v49;
    ++*(_QWORD *)a2;
    v45 = v43 + 1;
    v50 = v44;
    if ( 4 * v45 >= 3 * v42 )
    {
      v42 *= 2;
    }
    else if ( v42 - *(_DWORD *)(a2 + 20) - v45 > v42 >> 3 )
    {
LABEL_47:
      *(_DWORD *)(a2 + 16) = v45;
      if ( *v44 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v44 = v48[0];
      return v48[0];
    }
    sub_B02D10(a2, v42);
    sub_AFC4C0(a2, v48, &v50);
    v44 = v50;
    v45 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_47;
  }
  return v48[0];
}
