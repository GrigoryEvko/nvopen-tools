// Function: sub_970C30
// Address: 0x970c30
//
__int64 __fastcall sub_970C30(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  unsigned __int8 v4; // al
  unsigned int v8; // ebx
  __int128 v9; // rax
  signed __int64 v10; // rax
  __int64 v11; // rsi
  unsigned int v12; // ecx
  char *v13; // rdx
  unsigned int v14; // eax
  bool v15; // zf
  __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  unsigned __int8 *v18; // r14
  unsigned __int8 *v19; // rbx
  unsigned __int64 v20; // rcx
  _QWORD *v21; // rdx
  __int64 v22; // rdx
  _QWORD *v23; // rdx
  unsigned int v24; // eax
  __int64 v25; // r14
  __int128 v27; // rax
  unsigned int v28; // ebx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int8 *v32; // r15
  int v33; // edx
  __int64 v34; // rbx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  int v38; // edx
  int v39; // edx
  __int64 v40; // rax
  __int64 v41; // rsi
  unsigned __int64 v42; // rdx
  _QWORD *v43; // rax
  int v44; // r14d
  unsigned __int64 v45; // rcx
  _QWORD *v46; // rdx
  __int64 v47; // rsi
  _QWORD *v48; // rcx
  unsigned int v49; // eax
  __int64 v50; // r13
  __int64 v51; // [rsp+0h] [rbp-70h]
  unsigned int v52; // [rsp+8h] [rbp-68h]
  char v53; // [rsp+Ch] [rbp-64h]
  unsigned __int64 v54; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v55; // [rsp+18h] [rbp-58h]
  _OWORD v56[5]; // [rsp+20h] [rbp-50h] BYREF

  v4 = *(_BYTE *)(a2 + 8);
  if ( v4 == 18 )
    return 0;
  if ( v4 != 12 )
  {
    if ( v4 > 3u && v4 != 5 && (v4 & 0xFD) != 4 && v4 != 14 && v4 != 17 )
      return 0;
    *(_QWORD *)&v27 = sub_9208B0((__int64)a4, a2);
    v28 = v27;
    v56[0] = v27;
    v29 = sub_BD5C60(a1, *((_QWORD *)&v27 + 1), *((_QWORD *)&v27 + 1));
    v30 = sub_BCD140(v29, v28);
    v31 = sub_970C30(a1, v30, a3, a4);
    v32 = (unsigned __int8 *)v31;
    if ( !v31 )
      return 0;
    if ( (unsigned __int8)sub_AC30F0(v31) )
    {
      v34 = a2;
      if ( *(_BYTE *)(a2 + 8) != 10 )
        return sub_AD6530(a2);
    }
    else
    {
      v33 = *(unsigned __int8 *)(a2 + 8);
      if ( (unsigned int)(v33 - 17) <= 1 )
        LOBYTE(v33) = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
      v34 = a2;
      if ( (_BYTE)v33 == 14 )
        v34 = sub_AE4450(a4, a2);
    }
    v25 = sub_96E500(v32, v34, (__int64)a4);
    if ( !v25 )
      v25 = sub_96F860((__int64)v32, v34, a4, v35, v36, v37);
    v38 = *(unsigned __int8 *)(a2 + 8);
    if ( (unsigned int)(v38 - 17) <= 1 )
      LOBYTE(v38) = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
    if ( (_BYTE)v38 != 14 )
      return v25;
    if ( !(unsigned __int8)sub_AC30F0(v25) )
    {
      v39 = *(unsigned __int8 *)(a2 + 8);
      if ( (unsigned int)(v39 - 17) > 1 )
      {
        v40 = a2;
      }
      else
      {
        v40 = **(_QWORD **)(a2 + 16);
        LOBYTE(v39) = *(_BYTE *)(v40 + 8);
      }
      if ( (_BYTE)v39 == 14 && *(_BYTE *)(sub_AE2980(a4, *(_DWORD *)(v40 + 8) >> 8) + 16) )
        return 0;
      return sub_AD4C70(v25, a2, 0);
    }
    if ( *(_BYTE *)(a2 + 8) == 10 )
      return sub_AD4C70(v25, a2, 0);
    return sub_AD6530(a2);
  }
  v8 = (unsigned int)((*(_DWORD *)(a2 + 8) >> 8) + 7) >> 3;
  if ( v8 - 1 > 0x1F )
    return 0;
  if ( -(__int64)v8 < a3 )
  {
    v52 = v8 - 1;
    v51 = *(_QWORD *)(a1 + 8);
    v53 = sub_AE5020(a4, v51);
    *(_QWORD *)&v9 = sub_9208B0((__int64)a4, v51);
    v56[0] = v9;
    v10 = ((1LL << v53) + ((unsigned __int64)(v9 + 7) >> 3) - 1) >> v53 << v53;
    if ( BYTE8(v56[0]) )
      return 0;
    if ( v10 > a3 )
    {
      memset(v56, 0, 32);
      if ( a3 < 0 )
      {
        v12 = v8 + a3;
        v11 = 0;
        v13 = (char *)v56 - a3;
      }
      else
      {
        v11 = a3;
        v12 = v8;
        v13 = (char *)v56;
      }
      if ( (unsigned __int8)sub_9704C0((unsigned __int8 *)a1, v11, (__int64)v13, v12, a4) )
      {
        v55 = *(_DWORD *)(a2 + 8) >> 8;
        v14 = v55;
        if ( v55 > 0x40 )
        {
          sub_C43690(&v54, 0, 0);
          if ( *a4 )
          {
            v14 = v55;
            v16 = LOBYTE(v56[0]);
            if ( v55 > 0x40 )
            {
              *(_QWORD *)v54 = LOBYTE(v56[0]);
              memset((void *)(v54 + 8), 0, 8 * (unsigned int)(((unsigned __int64)v55 + 63) >> 6) - 8);
              goto LABEL_16;
            }
LABEL_13:
            v17 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v14) & v16;
            if ( !v14 )
              v17 = 0;
            v54 = v17;
LABEL_16:
            if ( v8 != 1 )
            {
              v18 = (unsigned __int8 *)v56 + 1;
              v19 = (unsigned __int8 *)v56 + v8;
              while ( 1 )
              {
                v24 = v55;
                if ( v55 <= 0x40 )
                  break;
                sub_C47690(&v54, 8);
                v24 = v55;
                v22 = *v18;
                if ( v55 <= 0x40 )
                {
                  v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v55;
LABEL_22:
                  v23 = (_QWORD *)(v20 & (v54 | v22));
                  if ( !v24 )
                    v23 = 0;
                  v54 = (unsigned __int64)v23;
                  goto LABEL_25;
                }
                *(_QWORD *)v54 |= v22;
LABEL_25:
                if ( ++v18 == v19 )
                  goto LABEL_76;
              }
              v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v55;
              v21 = 0;
              if ( v55 != 8 )
              {
                v21 = (_QWORD *)(v20 & (v54 << 8));
                if ( !v55 )
                  v21 = 0;
              }
              v54 = (unsigned __int64)v21;
              v22 = *v18;
              goto LABEL_22;
            }
            goto LABEL_76;
          }
          v14 = v55;
          v41 = *((unsigned __int8 *)v56 + v52);
          if ( v55 > 0x40 )
          {
            *(_QWORD *)v54 = v41;
            memset((void *)(v54 + 8), 0, 8 * (unsigned int)(((unsigned __int64)v55 + 63) >> 6) - 8);
            goto LABEL_62;
          }
        }
        else
        {
          v15 = *a4 == 0;
          v54 = 0;
          if ( !v15 )
          {
            v16 = LOBYTE(v56[0]);
            goto LABEL_13;
          }
          v41 = *((unsigned __int8 *)v56 + v52);
        }
        v42 = v41 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v14);
        v15 = v14 == 0;
        v43 = 0;
        if ( !v15 )
          v43 = (_QWORD *)v42;
        v54 = (unsigned __int64)v43;
LABEL_62:
        v44 = 2;
        if ( v8 == 1 )
        {
LABEL_76:
          v25 = sub_ACCFD0(*(_QWORD *)a2, &v54);
          if ( v55 > 0x40 )
          {
            if ( v54 )
              j_j___libc_free_0_0(v54);
          }
          return v25;
        }
        while ( 1 )
        {
          v49 = v55;
          v50 = v8 - v44;
          if ( v55 <= 0x40 )
          {
            v45 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v55;
            v46 = 0;
            if ( v55 != 8 )
            {
              v46 = (_QWORD *)(v45 & (v54 << 8));
              if ( !v55 )
                v46 = 0;
            }
            v54 = (unsigned __int64)v46;
            v47 = *((unsigned __int8 *)v56 + v50);
          }
          else
          {
            sub_C47690(&v54, 8);
            v49 = v55;
            v47 = *((unsigned __int8 *)v56 + v50);
            if ( v55 > 0x40 )
            {
              *(_QWORD *)v54 |= v47;
              goto LABEL_71;
            }
            v45 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v55;
          }
          v48 = (_QWORD *)((v54 | v47) & v45);
          if ( !v49 )
            v48 = 0;
          v54 = (unsigned __int64)v48;
LABEL_71:
          if ( v44 == v8 )
            goto LABEL_76;
          ++v44;
        }
      }
      return 0;
    }
  }
  return sub_ACADE0(a2);
}
