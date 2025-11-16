// Function: sub_2C895C0
// Address: 0x2c895c0
//
__int64 __fastcall sub_2C895C0(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  _QWORD *i; // r13
  _QWORD *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned int *v10; // rax
  int v11; // ecx
  unsigned int *v12; // rdx
  unsigned __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // r9
  unsigned __int8 v16; // al
  _BYTE *v17; // rdx
  _BYTE **v18; // r12
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rdx
  _BYTE **v25; // rdx
  __int64 v26; // rsi
  unsigned __int64 v27; // rsi
  unsigned __int64 v28; // rbx
  unsigned __int8 *v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // [rsp+8h] [rbp-148h]
  __int64 v35; // [rsp+10h] [rbp-140h]
  __int64 v36; // [rsp+18h] [rbp-138h]
  char v37; // [rsp+28h] [rbp-128h]
  __int64 v38; // [rsp+28h] [rbp-128h]
  __int64 v39; // [rsp+38h] [rbp-118h]
  unsigned int v40; // [rsp+58h] [rbp-F8h]
  __int64 v41[4]; // [rsp+60h] [rbp-F0h] BYREF
  char v42; // [rsp+80h] [rbp-D0h]
  char v43; // [rsp+81h] [rbp-CFh]
  unsigned int *v44; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+98h] [rbp-B8h]
  _BYTE v46[32]; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+C0h] [rbp-90h]
  _QWORD *v48; // [rsp+C8h] [rbp-88h]
  __int16 v49; // [rsp+D0h] [rbp-80h]
  __int64 v50; // [rsp+D8h] [rbp-78h]
  void **v51; // [rsp+E0h] [rbp-70h]
  _QWORD *v52; // [rsp+E8h] [rbp-68h]
  __int64 v53; // [rsp+F0h] [rbp-60h]
  int v54; // [rsp+F8h] [rbp-58h]
  __int16 v55; // [rsp+FCh] [rbp-54h]
  char v56; // [rsp+FEh] [rbp-52h]
  __int64 v57; // [rsp+100h] [rbp-50h]
  __int64 v58; // [rsp+108h] [rbp-48h]
  void *v59; // [rsp+110h] [rbp-40h] BYREF
  _QWORD v60[7]; // [rsp+118h] [rbp-38h] BYREF

  result = *(unsigned int *)(a1 + 8);
  if ( !(_DWORD)result )
    return result;
  v39 = 0;
  v36 = 8 * result;
  do
  {
    for ( i = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + v39) + 48LL) & 0xFFFFFFFFFFFFFFF8LL); ; i = (_QWORD *)v13 )
    {
      if ( !i )
      {
        v1 = sub_BD5C60(0);
        v52 = v60;
        v50 = v1;
        v44 = (unsigned int *)v46;
        v51 = &v59;
        v45 = 0x200000000LL;
        v53 = 0;
        v59 = &unk_49DA100;
        v54 = 0;
        v55 = 512;
        v56 = 7;
        v57 = 0;
        v58 = 0;
        v47 = 0;
        v48 = 0;
        v49 = 0;
        v60[0] = &unk_49DA0B0;
        BUG();
      }
      v4 = i - 3;
      v5 = sub_BD5C60((__int64)(i - 3));
      v51 = &v59;
      v50 = v5;
      v52 = v60;
      v59 = &unk_49DA100;
      v47 = 0;
      v53 = 0;
      v48 = 0;
      v54 = 0;
      v55 = 512;
      v56 = 7;
      v57 = 0;
      v58 = 0;
      v49 = 0;
      v60[0] = &unk_49DA0B0;
      v6 = i[2];
      v44 = (unsigned int *)v46;
      v45 = 0x200000000LL;
      v47 = v6;
      v48 = i;
      v7 = *(_QWORD *)sub_B46C60((__int64)(i - 3));
      v41[0] = v7;
      if ( !v7 )
        goto LABEL_24;
      sub_B96E90((__int64)v41, v7, 1);
      v9 = v41[0];
      if ( v41[0] )
      {
        v10 = v44;
        v11 = v45;
        v12 = &v44[4 * (unsigned int)v45];
        if ( v44 != v12 )
        {
          while ( *v10 )
          {
            v10 += 4;
            if ( v12 == v10 )
              goto LABEL_18;
          }
          *((_QWORD *)v10 + 1) = v41[0];
LABEL_12:
          sub_B91220((__int64)v41, v9);
          goto LABEL_13;
        }
LABEL_18:
        if ( (unsigned int)v45 >= (unsigned __int64)HIDWORD(v45) )
        {
          v27 = (unsigned int)v45 + 1LL;
          v28 = v35 & 0xFFFFFFFF00000000LL;
          v35 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v45) < v27 )
          {
            v38 = v41[0];
            sub_C8D5F0((__int64)&v44, v46, v27, 0x10u, v8, v41[0]);
            v9 = v38;
            v12 = &v44[4 * (unsigned int)v45];
          }
          *(_QWORD *)v12 = v28;
          *((_QWORD *)v12 + 1) = v9;
          v9 = v41[0];
          LODWORD(v45) = v45 + 1;
        }
        else
        {
          if ( v12 )
          {
            *v12 = 0;
            *((_QWORD *)v12 + 1) = v9;
            v11 = v45;
            v9 = v41[0];
          }
          LODWORD(v45) = v11 + 1;
        }
      }
      else
      {
LABEL_24:
        sub_93FB40((__int64)&v44, 0);
        v9 = v41[0];
      }
      if ( v9 )
        goto LABEL_12;
LABEL_13:
      if ( i == *(_QWORD **)(*(_QWORD *)(*(_QWORD *)a1 + v39) + 56LL) )
      {
        if ( *((_BYTE *)i - 24) != 45 )
          break;
        v37 = 1;
        v13 = (unsigned __int64)i;
      }
      else
      {
        v13 = *i & 0xFFFFFFFFFFFFFFF8LL;
        if ( *((_BYTE *)i - 24) != 45 )
          goto LABEL_15;
        v37 = 0;
      }
      if ( (*((_BYTE *)i - 17) & 0x40) != 0 )
      {
        v14 = *(i - 4);
        v15 = *(_QWORD *)(v14 + 32);
        v16 = *(_BYTE *)v15;
        if ( *(_BYTE *)v15 <= 0x15u )
          goto LABEL_29;
      }
      else
      {
        v14 = (__int64)&v4[-4 * (*((_DWORD *)i - 5) & 0x7FFFFFF)];
        v15 = *(_QWORD *)(v14 + 32);
        v16 = *(_BYTE *)v15;
        if ( *(_BYTE *)v15 <= 0x15u )
        {
LABEL_29:
          v43 = 1;
          v41[0] = (__int64)"conv2Add";
          v42 = 3;
          v17 = (_BYTE *)sub_AAAFF0(12, (unsigned __int8 *)v15, v14, v39, v8);
          if ( (*((_BYTE *)i - 17) & 0x40) != 0 )
            v18 = (_BYTE **)*(i - 4);
          else
            v18 = (_BYTE **)&v4[-4 * (*((_DWORD *)i - 5) & 0x7FFFFFF)];
          sub_92A220(&v44, *v18, v17, v40, (__int64)v41, 0);
          goto LABEL_32;
        }
      }
      if ( v16 > 0x1Cu )
      {
        v19 = *(_QWORD *)(v15 + 16);
        if ( v19 )
        {
          if ( !*(_QWORD *)(v19 + 8) && v16 == 47 )
          {
            if ( (*(_BYTE *)(v15 + 7) & 0x40) != 0 )
              v20 = *(_QWORD *)(v15 - 8);
            else
              v20 = v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF);
            v34 = v15;
            if ( **(_BYTE **)v20 > 0x15u )
            {
              v29 = *(unsigned __int8 **)(v20 + 32);
              if ( *v29 > 0x15u )
                goto LABEL_32;
              v30 = sub_AAAFF0(12, v29, v20, v39, v8);
              if ( (*(_BYTE *)(v34 + 7) & 0x40) != 0 )
                v31 = *(_QWORD *)(v34 - 8);
              else
                v31 = v34 - 32LL * (*(_DWORD *)(v34 + 4) & 0x7FFFFFF);
              if ( *(_QWORD *)(v31 + 32) )
              {
                v32 = *(_QWORD *)(v31 + 40);
                **(_QWORD **)(v31 + 48) = v32;
                if ( v32 )
                  *(_QWORD *)(v32 + 16) = *(_QWORD *)(v31 + 48);
              }
              *(_QWORD *)(v31 + 32) = v30;
              if ( v30 )
              {
                v33 = *(_QWORD *)(v30 + 16);
                *(_QWORD *)(v31 + 40) = v33;
                if ( v33 )
                  *(_QWORD *)(v33 + 16) = v31 + 40;
                *(_QWORD *)(v31 + 48) = v30 + 16;
                *(_QWORD *)(v30 + 16) = v31 + 32;
              }
            }
            else
            {
              v21 = sub_AAAFF0(12, *(unsigned __int8 **)v20, v20, v39, v8);
              if ( (*(_BYTE *)(v34 + 7) & 0x40) != 0 )
                v22 = *(_QWORD *)(v34 - 8);
              else
                v22 = v34 - 32LL * (*(_DWORD *)(v34 + 4) & 0x7FFFFFF);
              if ( *(_QWORD *)v22 )
              {
                v23 = *(_QWORD *)(v22 + 8);
                **(_QWORD **)(v22 + 16) = v23;
                if ( v23 )
                  *(_QWORD *)(v23 + 16) = *(_QWORD *)(v22 + 16);
              }
              *(_QWORD *)v22 = v21;
              if ( v21 )
              {
                v24 = *(_QWORD *)(v21 + 16);
                *(_QWORD *)(v22 + 8) = v24;
                if ( v24 )
                  *(_QWORD *)(v24 + 16) = v22 + 8;
                *(_QWORD *)(v22 + 16) = v21 + 16;
                *(_QWORD *)(v21 + 16) = v22;
              }
            }
            v43 = 1;
            v41[0] = (__int64)"conv2Add";
            v42 = 3;
            if ( (*((_BYTE *)i - 17) & 0x40) != 0 )
              v25 = (_BYTE **)*(i - 4);
            else
              v25 = (_BYTE **)&v4[-4 * (*((_DWORD *)i - 5) & 0x7FFFFFF)];
            v26 = sub_92A220(&v44, *v25, v25[4], v40, (__int64)v41, 0);
            if ( v26 )
            {
              sub_BD84D0((__int64)(i - 3), v26);
              sub_B43D60(i - 3);
            }
          }
        }
      }
LABEL_32:
      if ( v37 )
        break;
LABEL_15:
      nullsub_61();
      v59 = &unk_49DA100;
      nullsub_63();
      if ( v44 != (unsigned int *)v46 )
        _libc_free((unsigned __int64)v44);
    }
    nullsub_61();
    v59 = &unk_49DA100;
    nullsub_63();
    if ( v44 != (unsigned int *)v46 )
      _libc_free((unsigned __int64)v44);
    v39 += 8;
    result = v39;
  }
  while ( v36 != v39 );
  return result;
}
