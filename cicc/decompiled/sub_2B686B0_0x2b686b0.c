// Function: sub_2B686B0
// Address: 0x2b686b0
//
__int64 __fastcall sub_2B686B0(__int64 a1, _BYTE *a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // r9
  __int64 v7; // rdx
  __int64 v8; // rdi
  char v9; // r11
  __int64 v10; // r10
  char v11; // r8
  char v12; // al
  unsigned int v13; // eax
  unsigned int v14; // r12d
  __int64 v15; // r13
  unsigned int v16; // r15d
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rbx
  _BYTE *v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  char v25; // al
  bool v26; // al
  __int64 *v28; // rdx
  __int64 v29; // rax
  unsigned int v30; // [rsp+10h] [rbp-90h]
  int v31; // [rsp+14h] [rbp-8Ch]
  __int64 v32; // [rsp+18h] [rbp-88h]
  __int64 v33; // [rsp+18h] [rbp-88h]
  char v34; // [rsp+18h] [rbp-88h]
  char v35; // [rsp+20h] [rbp-80h]
  __int64 v36; // [rsp+20h] [rbp-80h]
  __int64 v37; // [rsp+20h] [rbp-80h]
  __int64 v38; // [rsp+28h] [rbp-78h]
  char v39; // [rsp+28h] [rbp-78h]
  char v40; // [rsp+28h] [rbp-78h]
  char v41; // [rsp+30h] [rbp-70h]
  __int64 v42; // [rsp+30h] [rbp-70h]
  __int64 v43; // [rsp+30h] [rbp-70h]
  char v44; // [rsp+38h] [rbp-68h]
  __int64 v45; // [rsp+38h] [rbp-68h]
  char v46; // [rsp+38h] [rbp-68h]
  __int64 v47; // [rsp+38h] [rbp-68h]
  __int64 v48; // [rsp+40h] [rbp-60h]
  unsigned int v49; // [rsp+40h] [rbp-60h]
  unsigned int v50; // [rsp+48h] [rbp-58h]
  __int64 v51[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v52[8]; // [rsp+60h] [rbp-40h] BYREF

  v4 = a3;
  v5 = (__int64)a2;
  v7 = *(_QWORD *)a1;
  v31 = a4;
  if ( *a2 == 61 && (v50 = *(_DWORD *)(v7 + 8), v50 == 2) )
  {
    if ( *(_DWORD *)(a1 + 208) == 2 )
      return 0;
    v8 = *(_QWORD *)(a1 + 248);
    v9 = 0;
    v10 = 16LL * (unsigned int)a4;
    v11 = *(_BYTE *)(*(_QWORD *)(v7 + 48 * v4) + v10 + 8);
    if ( !v8 )
      goto LABEL_5;
  }
  else
  {
    v8 = *(_QWORD *)(a1 + 248);
    v9 = 0;
    v10 = 16LL * (unsigned int)a4;
    v11 = *(_BYTE *)(*(_QWORD *)(v7 + 48 * v4) + v10 + 8);
    if ( !v8 )
      goto LABEL_4;
  }
  v44 = v11;
  v48 = v10;
  v12 = sub_D48480(v8, (__int64)a2, v7, a4);
  v7 = *(_QWORD *)a1;
  v11 = v44;
  v10 = v48;
  v5 = (__int64)a2;
  v9 = v12;
LABEL_4:
  v13 = *(_DWORD *)(v7 + 8);
  v50 = v13;
  if ( v13 )
  {
LABEL_5:
    v49 = 0;
    v30 = 0;
    while ( 1 )
    {
      if ( v31 == v49 )
        goto LABEL_26;
      if ( !*(_DWORD *)(a1 + 208) )
        return 0;
      v14 = 1;
      v15 = 0;
      v16 = *(_DWORD *)(a1 + 208);
      v17 = 16LL * v49;
      while ( 1 )
      {
        v18 = *(_QWORD *)a1;
        v19 = *(_QWORD *)(*(_QWORD *)a1 + v15);
        v20 = v19 + v17;
        if ( *(_BYTE *)(v19 + v17 + 8) != v11 || *(_BYTE *)(v20 + 9) )
          goto LABEL_12;
        LODWORD(v7) = **(unsigned __int8 **)(v19 + v10);
        v21 = *(_BYTE **)v20;
        if ( *(_QWORD *)v20 == v5 )
          goto LABEL_25;
        if ( (unsigned __int8)v7 <= 0x15u )
          break;
        if ( v50 <= 2 )
        {
          if ( v50 != 2 )
            break;
          v51[0] = v5;
          v34 = v11;
          v37 = v10;
          v40 = v9;
          v43 = v5;
          v47 = v17;
          v28 = *(__int64 **)(a1 + 216);
          v51[1] = *(_QWORD *)(*(_QWORD *)(v18 + 48LL * (v14 % v16)) + v17);
          v29 = sub_2B5F980(v51, 2u, v28);
          v17 = v47;
          v5 = v43;
          v9 = v40;
          v10 = v37;
          v11 = v34;
          if ( v29 )
          {
            if ( v7 )
              break;
          }
          v21 = *(_BYTE **)v20;
        }
        if ( *v21 <= 0x15u )
          goto LABEL_25;
        if ( v9 )
          goto LABEL_21;
LABEL_12:
        v15 += 48;
        if ( v16 == v14 )
          return 0;
        ++v14;
      }
      if ( !v9 || **(_BYTE **)v20 <= 0x15u )
        goto LABEL_12;
LABEL_21:
      v52[0] = v5;
      v22 = *(__int64 **)(a1 + 216);
      v32 = v17;
      v35 = v11;
      v38 = v10;
      v41 = v9;
      v45 = v5;
      v52[1] = *(_QWORD *)v20;
      v23 = sub_2B5F980(v52, 2u, v22);
      v5 = v45;
      v9 = v41;
      v10 = v38;
      v11 = v35;
      v17 = v32;
      if ( v23 && v24 )
        goto LABEL_12;
      v33 = v45;
      v36 = v17;
      v39 = v11;
      v42 = v10;
      v46 = v9;
      v25 = sub_D48480(*(_QWORD *)(a1 + 248), *(_QWORD *)v20, v24, v17);
      v9 = v46;
      v10 = v42;
      v11 = v39;
      v17 = v36;
      v5 = v33;
      if ( !v25 )
        goto LABEL_12;
      v21 = *(_BYTE **)v20;
LABEL_25:
      v26 = v21 == (_BYTE *)v5;
      *(_BYTE *)(v20 + 9) = v26;
      v30 += v26;
LABEL_26:
      if ( ++v49 == v50 )
      {
        LOBYTE(v7) = v30 > 1;
        v13 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
        v50 = v13;
        goto LABEL_28;
      }
    }
  }
  LODWORD(v7) = 0;
LABEL_28:
  LOBYTE(v13) = v50 == 2;
  return (unsigned int)v7 | v13;
}
