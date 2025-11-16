// Function: sub_12684A0
// Address: 0x12684a0
//
__int64 __fastcall sub_12684A0(__int64 a1)
{
  const char *v1; // r14
  size_t v2; // rbx
  __int64 v3; // rax
  _QWORD *v4; // r13
  size_t v5; // rdx
  char *v6; // r12
  _QWORD *v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rbx
  __int64 v14; // rcx
  _DWORD *v15; // r8
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 result; // rax
  size_t v21; // rax
  char v22; // dl
  char *v23; // rax
  __int64 i; // rdx
  __int64 v25; // r8
  char *v26; // rax
  char *v27; // rax
  _DWORD *v28; // [rsp+0h] [rbp-360h]
  _DWORD *v29; // [rsp+0h] [rbp-360h]
  void (__fastcall *v30)(__int64, __int64, _QWORD); // [rsp+8h] [rbp-358h]
  __int64 v31; // [rsp+8h] [rbp-358h]
  __int64 v32; // [rsp+8h] [rbp-358h]
  __int64 v33; // [rsp+8h] [rbp-358h]
  _QWORD *v34; // [rsp+10h] [rbp-350h] BYREF
  __int64 v35; // [rsp+18h] [rbp-348h]
  _QWORD v36[2]; // [rsp+20h] [rbp-340h] BYREF
  _BYTE v37[816]; // [rsp+30h] [rbp-330h] BYREF

  v34 = v36;
  v35 = 0;
  LOBYTE(v36[0]) = 0;
  sub_16033C0(a1, unk_4D04618 != 0);
  unk_4F92C78 = 0;
  v1 = (const char *)qword_4D046D8;
  if ( qword_4D046D8 )
  {
    if ( *qword_4D046D8 )
    {
      v21 = (size_t)&v1[strlen((const char *)qword_4D046D8) - 1];
      if ( v1 != (const char *)v21 )
      {
        while ( 1 )
        {
          v22 = *(_BYTE *)(v21 - 1);
          if ( v22 == 92 || v22 == 47 )
            break;
          if ( v1 == (const char *)--v21 )
            goto LABEL_3;
        }
        v1 = (const char *)v21;
      }
    }
  }
  else
  {
    v1 = "moduleOutput";
  }
LABEL_3:
  v2 = strlen(v1);
  v3 = sub_22077B0(736);
  v4 = (_QWORD *)v3;
  if ( v3 )
    sub_1631D60(v3, v1, v2, a1);
  if ( *(_QWORD *)&dword_4F06A68 == 8 )
  {
    if ( unk_4D0461C )
      sub_2241130(
        &v34,
        0,
        v35,
        "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:1"
        "28-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64",
        167);
    else
      sub_2241130(
        &v34,
        0,
        v35,
        "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16"
        "-v32:32:32-v64:64:64-v128:128:128-n16:32:64",
        155);
    sub_1268290(v4, "nvptx64-nvidia-cuda", 0x13u);
    if ( unk_4D04630 )
      goto LABEL_9;
  }
  else
  {
    sub_2241130(
      &v34,
      0,
      v35,
      "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v"
      "32:32:32-v64:64:64-v128:128:128-n16:32:64",
      155);
    sub_1268290(v4, "nvptx-nvidia-cuda", 0x11u);
    if ( unk_4D04630 )
    {
LABEL_9:
      v5 = 0;
      v6 = off_4CD49B0;
      if ( off_4CD49B0 )
        v5 = strlen(off_4CD49B0);
      sub_1268290(v4, v6, v5);
      goto LABEL_12;
    }
  }
  sub_1632B30(v4, v34, v35);
LABEL_12:
  v7 = v34;
  v8 = v35;
  v9 = sub_22077B0(456);
  v10 = v9;
  if ( v9 )
  {
    *(_QWORD *)(v9 + 200) = 0;
    *(_QWORD *)(v9 + 48) = v9 + 64;
    *(_QWORD *)(v9 + 192) = v9 + 208;
    *(_QWORD *)(v9 + 24) = v9 + 40;
    *(_QWORD *)(v9 + 224) = v9 + 240;
    *(_QWORD *)(v9 + 56) = 0x1000000000LL;
    *(_QWORD *)(v9 + 408) = v9 + 424;
    *(_QWORD *)(v9 + 32) = 0x800000000LL;
    *(_BYTE *)(v9 + 208) = 0;
    *(_QWORD *)(v9 + 232) = 0x800000000LL;
    *(_QWORD *)(v9 + 400) = 0;
    *(_QWORD *)(v9 + 416) = 0x800000000LL;
    sub_15A9300(v9, v7, v8);
  }
  sub_1269840(v37, v4, v10);
  if ( unk_4D04610 )
    goto LABEL_42;
  v11 = qword_4F07288;
  v12 = *(_QWORD *)(qword_4F07288 + 104);
  if ( v12 )
  {
    do
    {
      while ( (unsigned int)sub_736DD0(v12) )
      {
        v12 = *(_QWORD *)(v12 + 112);
        if ( !v12 )
          goto LABEL_20;
      }
      sub_127C6B0(v12);
      v12 = *(_QWORD *)(v12 + 112);
    }
    while ( v12 );
LABEL_20:
    v11 = qword_4F07288;
  }
  v13 = *(_QWORD *)(v11 + 112);
  if ( !v13 )
    goto LABEL_33;
  do
  {
    if ( (*(_BYTE *)(v13 + 170) & 0x60) == 0
      && *(_BYTE *)(v13 + 177) != 5
      && ((*(_BYTE *)(v13 + 176) & 0x40) == 0 || *(_BYTE *)(*(_QWORD *)(v13 + 120) + 140LL) != 14)
      && (unk_4D04630
       || sub_5E3770(v13)
       && ((unsigned __int8)sub_127BF90(*(_QWORD *)(v13 + 120)) || (*(_BYTE *)(v13 + 156) & 1) != 0)
       && (*(_BYTE *)(v13 - 8) & 0x10) == 0) )
    {
      for ( i = *(_QWORD *)(v13 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( (*(_BYTE *)(v13 + 156) & 2) == 0
        || *(_QWORD *)(i + 128)
        || (v33 = i, !sub_8D3410(i))
        || *(_QWORD *)(v33 + 176)
        || (*(_BYTE *)(v33 + 169) & 0x20) != 0 )
      {
        if ( !HIDWORD(qword_4D045BC) )
        {
          if ( *(_BYTE *)(v13 + 136) != 1 )
          {
            v25 = v13 + 64;
LABEL_73:
            *(_BYTE *)(v13 + 136) = 2;
LABEL_74:
            sub_127C770(v25);
            sub_1277390(v37, v13, 0);
            goto LABEL_31;
          }
          if ( !sub_8D23B0(*(_QWORD *)(v13 + 120)) )
          {
            v25 = v13 + 64;
            if ( *(_BYTE *)(v13 + 136) == 1 && (*(_BYTE *)(v13 - 8) & 0x10) == 0 && (*(_BYTE *)(v13 + 169) & 0x30) != 0 )
            {
              v26 = (char *)sub_127B360(v13);
              v27 = sub_693CD0(v26);
              sub_684B10(0xDADu, (_DWORD *)(v13 + 64), (__int64)v27);
              v25 = v13 + 64;
            }
            goto LABEL_73;
          }
        }
      }
      if ( *(_BYTE *)(v13 + 136) != 1 )
      {
        v25 = v13 + 64;
        goto LABEL_74;
      }
    }
LABEL_31:
    v13 = *(_QWORD *)(v13 + 112);
  }
  while ( v13 );
  v11 = qword_4F07288;
LABEL_33:
  v14 = *(_QWORD *)(v11 + 144);
  if ( v14 )
  {
    v15 = (_DWORD *)&qword_4D045BC + 1;
    do
    {
      if ( !*v15 )
      {
        v16 = *(_BYTE *)(v14 + 198);
        if ( (v16 & 0x20) != 0 && (*(_DWORD *)(v14 + 192) & 0x8000400) == 0 && (*(_BYTE *)(v14 + 195) & 1) != 0 )
        {
          if ( *(_DWORD *)(v14 + 160) )
          {
            if ( (v16 & 0x10) != 0 )
              goto LABEL_61;
            goto LABEL_41;
          }
          if ( !unk_4D0471C )
            goto LABEL_41;
          v28 = v15;
          v31 = v14;
          v23 = sub_8258E0(v14, 0);
          sub_684B10(0xE99u, (_DWORD *)(v31 + 64), (__int64)v23);
          v15 = v28;
          v14 = v31;
        }
      }
      if ( *(_DWORD *)(v14 + 160) && (*(_BYTE *)(v14 + 198) & 0x10) != 0 )
      {
LABEL_61:
        if ( (unk_4D04630 || (*(_BYTE *)(v14 + 203) & 4) != 0 && (*(_BYTE *)(v14 - 8) & 0x10) == 0)
          && (*(_DWORD *)(v14 + 192) & 0x8000400) == 0 )
        {
          v29 = v15;
          v32 = v14;
          sub_1276140(v37, v14);
          v15 = v29;
          v14 = v32;
        }
      }
LABEL_41:
      v14 = *(_QWORD *)(v14 + 112);
    }
    while ( v14 );
  }
LABEL_42:
  sub_1274F60(v37);
  v17 = sub_22077B0(24);
  v18 = v17;
  if ( v17 )
    sub_1611EE0(v17);
  v30 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v18 + 16LL);
  v19 = sub_1654860(0);
  v30(v18, v19, 0);
  if ( (unsigned __int8)sub_1619BD0(v18, v4) )
    sub_127B630("there was an error in verifying the lgenfe output!", 0);
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
  if ( v10 )
  {
    sub_15A93E0(v10);
    j_j___libc_free_0(v10, 456);
  }
  unk_4F92C78 = v4;
  result = sub_1269AB0(v37);
  if ( v34 != v36 )
    return j_j___libc_free_0(v34, v36[0] + 1LL);
  return result;
}
