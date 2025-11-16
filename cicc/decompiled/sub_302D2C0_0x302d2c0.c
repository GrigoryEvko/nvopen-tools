// Function: sub_302D2C0
// Address: 0x302d2c0
//
void __fastcall sub_302D2C0(__int64 a1)
{
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  __int64 v6; // rdx
  _BYTE *v7; // rdi
  __int64 v8; // r14
  __int64 *v9; // rdi
  __int64 v10; // rdx
  unsigned __int64 *v11; // rax
  int v12; // r9d
  __int64 v13; // rbx
  unsigned int v14; // eax
  __int64 v15; // r8
  __int64 v16; // r15
  __int64 *v17; // rdi
  __int64 v18; // rax
  unsigned __int8 v19; // dl
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // rax
  unsigned int v23; // r14d
  void (__fastcall *v24)(__int64, const char *, __int64, _QWORD, __int64); // rbx
  unsigned __int8 v25; // dl
  __int64 *v26; // rax
  unsigned __int8 v27; // dl
  __int64 *v28; // rax
  const char *v29; // rsi
  __int64 v30; // rdx
  __int64 *v31; // rdi
  int v32; // edx
  __int64 v33; // rbx
  unsigned int v34; // r9d
  unsigned int v35; // eax
  _QWORD *v36; // rdi
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rax
  _QWORD *v39; // rax
  __int64 v40; // rdx
  _QWORD *i; // rdx
  _QWORD *v42; // rax
  __int64 v43; // [rsp+8h] [rbp-188h]
  int v44; // [rsp+14h] [rbp-17Ch]
  unsigned __int64 v45[2]; // [rsp+20h] [rbp-170h] BYREF
  __int64 v46; // [rsp+30h] [rbp-160h] BYREF
  char v47; // [rsp+40h] [rbp-150h]
  unsigned __int64 *v48; // [rsp+50h] [rbp-140h] BYREF
  __int64 v49; // [rsp+58h] [rbp-138h]
  __int16 v50; // [rsp+70h] [rbp-120h]
  _QWORD v51[3]; // [rsp+80h] [rbp-110h] BYREF
  _BYTE *v52; // [rsp+98h] [rbp-F8h]
  _BYTE *v53; // [rsp+A0h] [rbp-F0h]
  __int64 v54; // [rsp+A8h] [rbp-E8h]
  unsigned __int64 *v55; // [rsp+B0h] [rbp-E0h]
  unsigned __int64 v56[3]; // [rsp+C0h] [rbp-D0h] BYREF
  _BYTE v57[184]; // [rsp+D8h] [rbp-B8h] BYREF

  v56[0] = (unsigned __int64)v57;
  v54 = 0x100000000LL;
  v51[0] = &unk_49DD288;
  v56[1] = 0;
  v56[2] = 128;
  v51[1] = 2;
  v51[2] = 0;
  v52 = 0;
  v53 = 0;
  v55 = v56;
  sub_CB5980((__int64)v51, 0, 0, 0);
  if ( !*(_BYTE *)(a1 + 1096) )
  {
    sub_302CFC0(a1, *(_QWORD *)(**(_QWORD **)(a1 + 232) + 40LL));
    *(_BYTE *)(a1 + 1096) = 1;
  }
  v5 = *(_QWORD *)(a1 + 232);
  v6 = *(_QWORD *)(v5 + 32);
  *(_QWORD *)(a1 + 1104) = v6;
  v7 = *(_BYTE **)v5;
  *(_QWORD *)(a1 + 1080) = *(_QWORD *)v5;
  if ( LOBYTE(qword_5036408[8]) )
  {
    v22 = sub_B92180((__int64)v7);
    if ( v22 )
    {
      v23 = *(_DWORD *)(v22 + 16);
      v24 = *(void (__fastcall **)(__int64, const char *, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 312LL);
      if ( *(_BYTE *)v22 == 16
        || ((v25 = *(_BYTE *)(v22 - 16), (v25 & 2) == 0)
          ? (v26 = (__int64 *)(v22 - 16 - 8LL * ((v25 >> 2) & 0xF)))
          : (v26 = *(__int64 **)(v22 - 32)),
            (v22 = *v26) != 0) )
      {
        v27 = *(_BYTE *)(v22 - 16);
        if ( (v27 & 2) != 0 )
          v28 = *(__int64 **)(v22 - 32);
        else
          v28 = (__int64 *)(v22 - 16 - 8LL * ((v27 >> 2) & 0xF));
        v29 = (const char *)*v28;
        if ( *v28 )
          v29 = (const char *)sub_B91420(*v28);
        else
          v30 = 0;
      }
      else
      {
        v30 = 0;
        v29 = byte_3F871B3;
      }
      v24(a1, v29, v30, v23, 1);
    }
    v7 = *(_BYTE **)(a1 + 1080);
  }
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 200) + 1280LL) == 1 )
  {
    sub_3022060(v7, (__int64)v51, v6, v2, v3, v4);
    v7 = *(_BYTE **)(a1 + 1080);
  }
  if ( (unsigned __int8)sub_CE9220((__int64)v7) )
  {
    sub_904010((__int64)v51, ".entry ");
    v8 = *(_QWORD *)(a1 + 1080);
    if ( (unsigned __int8)sub_CE9620(v8) )
      sub_3021B30(v8, (__int64)v51);
  }
  else
  {
    sub_904010((__int64)v51, ".func ");
    v21 = *(_QWORD *)(a1 + 1080);
    if ( (unsigned __int8)sub_CE9620(v21) )
      sub_3021B30(v21, (__int64)v51);
    sub_3022420(
      a1,
      **(_QWORD **)(*(_QWORD *)(**(_QWORD **)(a1 + 232) + 24LL) + 16LL),
      **(_QWORD **)(a1 + 232),
      (__int64)v51);
  }
  sub_EA12C0(*(_QWORD *)(a1 + 280), (__int64)v51, *(_BYTE **)(a1 + 208));
  sub_3024DF0(a1, *(_QWORD *)(*(_QWORD *)(a1 + 1080) + 24LL), *(_QWORD *)(a1 + 1080), (__int64)v51);
  if ( v52 == v53 )
    sub_CB6200((__int64)v51, (unsigned __int8 *)"\n", 1u);
  else
    *v53++ = 10;
  if ( (unsigned __int8)sub_CE9220(*(_QWORD *)(a1 + 1080)) )
    sub_3022E70(a1, *(_QWORD *)(a1 + 1080), (__int64)v51);
  sub_3022B50(a1, *(_QWORD *)(a1 + 1080), (__int64)v51);
  if ( (unsigned __int8)sub_307AAA0(*(_QWORD *)(a1 + 1080), *(_QWORD *)(a1 + 200)) )
    sub_904010((__int64)v51, ".noreturn");
  v9 = *(__int64 **)(a1 + 224);
  v10 = v55[1];
  v11 = (unsigned __int64 *)*v55;
  v50 = 261;
  v48 = v11;
  v49 = v10;
  sub_E99A90(v9, (__int64)&v48);
  v12 = *(_DWORD *)(a1 + 1128);
  ++*(_QWORD *)(a1 + 1112);
  if ( v12 || *(_DWORD *)(a1 + 1132) )
  {
    v13 = *(_QWORD *)(a1 + 1120);
    v14 = 4 * v12;
    v15 = 40LL * *(unsigned int *)(a1 + 1136);
    if ( (unsigned int)(4 * v12) < 0x40 )
      v14 = 64;
    v16 = v13 + v15;
    if ( *(_DWORD *)(a1 + 1136) <= v14 )
    {
      while ( v13 != v16 )
      {
        if ( *(_QWORD *)v13 != -4096 )
        {
          if ( *(_QWORD *)v13 != -8192 )
            sub_C7D6A0(*(_QWORD *)(v13 + 16), 8LL * *(unsigned int *)(v13 + 32), 4);
          *(_QWORD *)v13 = -4096;
        }
        v13 += 40;
      }
    }
    else
    {
      do
      {
        if ( *(_QWORD *)v13 != -8192 && *(_QWORD *)v13 != -4096 )
        {
          v43 = v15;
          v44 = v12;
          sub_C7D6A0(*(_QWORD *)(v13 + 16), 8LL * *(unsigned int *)(v13 + 32), 4);
          v15 = v43;
          v12 = v44;
        }
        v13 += 40;
      }
      while ( v13 != v16 );
      v32 = *(_DWORD *)(a1 + 1136);
      if ( v12 )
      {
        v33 = 64;
        v34 = v12 - 1;
        if ( v34 )
        {
          _BitScanReverse(&v35, v34);
          v33 = (unsigned int)(1 << (33 - (v35 ^ 0x1F)));
          if ( (int)v33 < 64 )
            v33 = 64;
        }
        v36 = *(_QWORD **)(a1 + 1120);
        if ( (_DWORD)v33 == v32 )
        {
          *(_QWORD *)(a1 + 1128) = 0;
          v42 = &v36[5 * v33];
          do
          {
            if ( v36 )
              *v36 = -4096;
            v36 += 5;
          }
          while ( v42 != v36 );
        }
        else
        {
          sub_C7D6A0((__int64)v36, v15, 8);
          v37 = ((((((((4 * (int)v33 / 3u + 1) | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v33 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 4)
                 | (((4 * (int)v33 / 3u + 1) | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v33 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 8)
               | (((((4 * (int)v33 / 3u + 1) | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v33 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v33 / 3u + 1) | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v33 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 16;
          v38 = (v37
               | (((((((4 * (int)v33 / 3u + 1) | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v33 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 4)
                 | (((4 * (int)v33 / 3u + 1) | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v33 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 8)
               | (((((4 * (int)v33 / 3u + 1) | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v33 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v33 / 3u + 1) | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v33 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v33 / 3u + 1) >> 1))
              + 1;
          *(_DWORD *)(a1 + 1136) = v38;
          v39 = (_QWORD *)sub_C7D670(40 * v38, 8);
          v40 = *(unsigned int *)(a1 + 1136);
          *(_QWORD *)(a1 + 1128) = 0;
          *(_QWORD *)(a1 + 1120) = v39;
          for ( i = &v39[5 * v40]; i != v39; v39 += 5 )
          {
            if ( v39 )
              *v39 = -4096;
          }
        }
        goto LABEL_28;
      }
      if ( v32 )
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 1120), v15, 8);
        *(_QWORD *)(a1 + 1120) = 0;
        *(_QWORD *)(a1 + 1128) = 0;
        *(_DWORD *)(a1 + 1136) = 0;
        goto LABEL_28;
      }
    }
    *(_QWORD *)(a1 + 1128) = 0;
  }
LABEL_28:
  v17 = *(__int64 **)(a1 + 224);
  v50 = 261;
  v48 = (unsigned __int64 *)"{\n";
  v49 = 2;
  sub_E99A90(v17, (__int64)&v48);
  sub_3022230((__int64)v45, *(_QWORD *)(a1 + 1080));
  if ( v47 )
  {
    v31 = *(__int64 **)(a1 + 224);
    v48 = v45;
    v50 = 260;
    sub_E99A90(v31, (__int64)&v48);
    if ( v47 )
    {
      v47 = 0;
      if ( (__int64 *)v45[0] != &v46 )
        j_j___libc_free_0(v45[0]);
    }
  }
  sub_302BBD0(a1, *(_QWORD *)(a1 + 232));
  v18 = sub_B92180(**(_QWORD **)(a1 + 232));
  if ( v18 )
  {
    v19 = *(_BYTE *)(v18 - 16);
    v20 = (v19 & 2) != 0 ? *(_QWORD *)(v18 - 32) : v18 - 16 - 8LL * ((v19 >> 2) & 0xF);
    if ( *(_DWORD *)(*(_QWORD *)(v20 + 40) + 32LL) != 3 )
      sub_31DB4A0(a1, *(_QWORD *)(a1 + 232));
  }
  v51[0] = &unk_49DD388;
  sub_CB5840((__int64)v51);
  if ( (_BYTE *)v56[0] != v57 )
    _libc_free(v56[0]);
}
