// Function: sub_3239180
// Address: 0x3239180
//
__int64 __fastcall sub_3239180(__int64 a1)
{
  __int64 *v2; // rdx
  __int64 *v3; // r15
  __int64 v4; // rax
  char v5; // al
  __int64 v6; // r14
  __int64 v7; // r9
  __int64 v8; // r12
  _QWORD *v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rax
  unsigned __int8 v14; // dl
  __int64 v15; // rcx
  unsigned int v16; // r10d
  unsigned __int64 v17; // r12
  int v18; // eax
  int v19; // r12d
  __int64 v20; // rax
  __int64 v21; // r13
  unsigned __int64 v22; // r14
  __int64 result; // rax
  __int64 *v24; // r14
  __int64 *j; // r12
  __int64 v26; // rbx
  __int64 v27; // r15
  __int64 v28; // rax
  bool v29; // zf
  __int64 v30; // rdx
  unsigned __int16 v31; // ax
  const void *v32; // rsi
  _BYTE *v33; // rdi
  size_t v34; // rdx
  __int64 v35; // [rsp+10h] [rbp-140h]
  unsigned int v36; // [rsp+10h] [rbp-140h]
  __int64 v37; // [rsp+18h] [rbp-138h]
  unsigned int v38; // [rsp+18h] [rbp-138h]
  unsigned int v39; // [rsp+20h] [rbp-130h]
  __int64 v40; // [rsp+20h] [rbp-130h]
  __int64 v41; // [rsp+20h] [rbp-130h]
  char v42; // [rsp+20h] [rbp-130h]
  _QWORD *v43; // [rsp+28h] [rbp-128h]
  __int64 *v44; // [rsp+38h] [rbp-118h]
  __int64 i; // [rsp+38h] [rbp-118h]
  unsigned __int64 v46; // [rsp+40h] [rbp-110h] BYREF
  int v47; // [rsp+48h] [rbp-108h]
  _BYTE *v48; // [rsp+50h] [rbp-100h] BYREF
  __int64 v49; // [rsp+58h] [rbp-F8h]
  _BYTE dest[136]; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v51; // [rsp+E8h] [rbp-68h]
  __int64 v52; // [rsp+F0h] [rbp-60h]
  __int64 v53; // [rsp+F8h] [rbp-58h]
  __int64 v54; // [rsp+100h] [rbp-50h]
  __int64 v55; // [rsp+108h] [rbp-48h]
  unsigned int v56; // [rsp+110h] [rbp-40h]

  v43 = (_QWORD *)sub_31DA6B0(*(_QWORD *)(a1 + 8));
  sub_32390D0(a1);
  sub_321F6F0(a1);
  v2 = *(__int64 **)(a1 + 656);
  v44 = &v2[2 * *(unsigned int *)(a1 + 664)];
  if ( v2 != v44 )
  {
    v3 = *(__int64 **)(a1 + 656);
    while ( 1 )
    {
      v6 = v3[1];
      if ( *(_DWORD *)(*(_QWORD *)(v6 + 80) + 32LL) != 3 )
        break;
LABEL_9:
      v3 += 2;
      if ( v44 == v3 )
        goto LABEL_48;
    }
    sub_3249500(v3[1]);
    v8 = *(_QWORD *)(v6 + 408);
    if ( v8 )
    {
      v9 = *(_QWORD **)(v6 + 40);
      if ( v9 && (*v9 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v10 = *(_QWORD *)(v6 + 408);
        v39 = (unsigned __int16)sub_3220AA0(a1) < 5u ? 8496 : 118;
        sub_3221260(a1, *(_QWORD *)(v6 + 80), v6);
        v11 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL);
        v35 = *(_QWORD *)(v11 + 1072);
        v37 = *(_QWORD *)(v11 + 1080);
        sub_324AD70(v6, v6 + 8, v39, v35, v37);
        sub_324AD70(v8, v8 + 8, v39, v35, v37);
        v40 = *(_QWORD *)(a1 + 8);
        sub_C7D030(&v48);
        v52 = v6;
        v53 = 0;
        v54 = 0;
        v55 = 0;
        v51 = v40;
        v56 = 0;
        v41 = sub_3734A30(&v48, v35, v37, v6 + 8);
        sub_C7D6A0(v54, 16LL * v56, 8);
        if ( (unsigned __int16)sub_3220AA0(a1) <= 4u )
        {
          LODWORD(v48) = 65543;
          sub_3249A20(v6, v6 + 16, 8497, v48, v41);
          LODWORD(v48) = 65543;
          sub_3249A20(v8, v8 + 16, 8497, v48, v41);
        }
        else
        {
          *(_QWORD *)(v6 + 736) = v41;
          *(_QWORD *)(v8 + 736) = v41;
        }
        if ( (unsigned __int16)sub_3220AA0(a1) <= 4u && *(_DWORD *)(a1 + 4024) )
          sub_324AC60(v8, v8 + 8, 8498, *(_QWORD *)(v43[20] + 16LL), *(_QWORD *)(v43[20] + 16LL));
        v12 = *(_DWORD *)(v6 + 480);
        if ( !v12 )
        {
LABEL_24:
          if ( !*(_DWORD *)(a1 + 4856) )
          {
LABEL_25:
            if ( (unsigned __int16)sub_3220AA0(a1) > 4u )
            {
              if ( *(_BYTE *)(v8 + 392) )
                sub_324AD20(v10);
              if ( *(_DWORD *)(a1 + 1296) && !*(_BYTE *)(a1 + 3769) )
                sub_324AC60(v10, v8 + 8, 140, *(_QWORD *)(a1 + 2776), *(_QWORD *)(v43[41] + 16LL));
            }
            v13 = *v3;
            v14 = *(_BYTE *)(*v3 - 16);
            if ( (v14 & 2) != 0 )
              v4 = *(_QWORD *)(v13 - 32);
            else
              v4 = v13 - 16 - 8LL * ((v14 >> 2) & 0xF);
            if ( *(_QWORD *)(v4 + 64) )
            {
              v5 = *(_BYTE *)(a1 + 3769);
              if ( *(_BYTE *)(a1 + 3692) )
              {
                if ( v5 )
                {
                  sub_324AB90(v6, v6 + 8, 121, *(_QWORD *)(v8 + 416), *(_QWORD *)(v43[37] + 16LL));
                }
                else
                {
                  v31 = sub_3220AA0(a1);
                  sub_324AC60(v10, v8 + 8, v31 < 5u ? 8473 : 121, *(_QWORD *)(v8 + 416), *(_QWORD *)(v43[22] + 16LL));
                }
              }
              else
              {
                v15 = *(_QWORD *)(v8 + 416);
                if ( v5 )
                  sub_324AB90(v6, v6 + 8, 67, v15, *(_QWORD *)(v43[36] + 16LL));
                else
                  sub_324AC60(v10, v8 + 8, 67, v15, *(_QWORD *)(v43[21] + 16LL));
              }
            }
            sub_3247F60(v10);
            goto LABEL_9;
          }
LABEL_40:
          sub_3737610(v8);
          goto LABEL_25;
        }
        v42 = 1;
        goto LABEL_21;
      }
      v10 = *(_QWORD *)(v6 + 408);
      sub_3221260(a1, *(_QWORD *)(v8 + 80), v10);
    }
    else
    {
      v10 = v6;
      v8 = v6;
    }
    v12 = *(_DWORD *)(v6 + 480);
    if ( !v12 )
      goto LABEL_38;
    v42 = 0;
LABEL_21:
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL) + 544LL) - 42) > 1 )
    {
      if ( v12 != 1 && *(_BYTE *)(a1 + 3688) )
      {
        LODWORD(v48) = 65537;
        sub_3249A20(v10, v8 + 16, 17, v48, 0);
      }
      else
      {
        *(_QWORD *)(v8 + 520) = **(_QWORD **)(v6 + 472);
      }
      v48 = dest;
      v49 = 0x200000000LL;
      v16 = *(_DWORD *)(v6 + 480);
      if ( v16 && &v48 != (_BYTE **)(v6 + 472) )
      {
        v32 = (const void *)(v6 + 488);
        if ( *(_QWORD *)(v6 + 472) == v6 + 488 )
        {
          v33 = dest;
          v34 = 16LL * v16;
          if ( v16 <= 2
            || (v36 = *(_DWORD *)(v6 + 480),
                sub_C8D5F0((__int64)&v48, dest, v16, 0x10u, (__int64)dest, v7),
                v33 = v48,
                v32 = *(const void **)(v6 + 472),
                v34 = 16LL * *(unsigned int *)(v6 + 480),
                v16 = v36,
                v34) )
          {
            v38 = v16;
            memcpy(v33, v32, v34);
            v16 = v38;
          }
          LODWORD(v49) = v16;
          *(_DWORD *)(v6 + 480) = 0;
        }
        else
        {
          v48 = *(_BYTE **)(v6 + 472);
          LODWORD(v49) = v16;
          HIDWORD(v49) = *(_DWORD *)(v6 + 484);
          *(_QWORD *)(v6 + 472) = v32;
          *(_QWORD *)(v6 + 480) = 0;
        }
      }
      sub_3739060(v8, v8 + 8, &v48);
      if ( v48 != dest )
        _libc_free((unsigned __int64)v48);
    }
    else
    {
      LODWORD(v48) = 65537;
      sub_3249A20(v10, v8 + 16, 17, v48, 0);
    }
    if ( v42 )
      goto LABEL_24;
LABEL_38:
    if ( (unsigned __int16)sub_3220AA0(a1) <= 4u || !*(_DWORD *)(a1 + 4856) )
      goto LABEL_25;
    goto LABEL_40;
  }
LABEL_48:
  v17 = sub_BA8DC0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL), (__int64)"llvm.dbg.cu", 11);
  v18 = 0;
  if ( v17 )
    v18 = sub_B91A00(v17);
  LODWORD(v49) = v18;
  v48 = (_BYTE *)v17;
  sub_BA95A0((__int64)&v48);
  v46 = v17;
  v47 = 0;
  sub_BA95A0((__int64)&v46);
  v19 = v49;
  LODWORD(v49) = v47;
  v48 = (_BYTE *)v46;
  if ( v47 != v19 )
  {
    do
    {
      v20 = sub_BA9580((__int64)&v48);
      if ( *(_QWORD *)(v20 + 24) )
        sub_3238860(a1, v20);
      LODWORD(v49) = v49 + 1;
      sub_BA95A0((__int64)&v48);
    }
    while ( (_DWORD)v49 != v19 );
  }
  sub_3244F70(a1 + 3080);
  if ( *(_BYTE *)(a1 + 3769) )
    sub_3244F70(a1 + 3776);
  v21 = *(_QWORD *)(a1 + 5016);
  v22 = (unsigned __int64)*(unsigned int *)(a1 + 5024) << 6;
  result = v21 + v22;
  for ( i = v21 + v22; i != v21; v21 += 64 )
  {
    v24 = *(__int64 **)(v21 + 40);
    for ( j = *(__int64 **)(v21 + 32); v24 != j; ++j )
    {
      v26 = *j;
      result = *(unsigned __int8 *)(*j + 16);
      if ( (_BYTE)result != 1 )
      {
        if ( (_BYTE)result )
          abort();
        v27 = *(_QWORD *)(v26 + 8);
        v28 = sub_37236D0(v27);
        v29 = *(_BYTE *)(v26 + 16) == 1;
        *(_QWORD *)(v26 + 24) = v28;
        *(_QWORD *)(v26 + 32) = v30;
        result = *(unsigned int *)(v27 + 16);
        if ( !v29 )
          *(_BYTE *)(v26 + 16) = 1;
        *(_QWORD *)(v26 + 8) = result;
      }
    }
  }
  return result;
}
