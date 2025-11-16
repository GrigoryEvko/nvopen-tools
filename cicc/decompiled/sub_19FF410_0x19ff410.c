// Function: sub_19FF410
// Address: 0x19ff410
//
__int64 __fastcall sub_19FF410(__int64 a1, __int64 *a2, double a3, double a4, double a5)
{
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // r12
  unsigned int v10; // eax
  __int64 v11; // r15
  __int64 v12; // rax
  char v13; // dl
  __int64 v14; // rcx
  unsigned int v15; // eax
  __int64 *v16; // rcx
  __int64 v17; // rsi
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // r15d
  __int64 v22; // rdi
  __int64 *v23; // r15
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // rsi
  unsigned __int8 *v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 *v31; // r15
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rsi
  __int64 v35; // rsi
  __int64 v36; // rdx
  unsigned __int8 *v37; // rsi
  unsigned __int8 *v38; // [rsp+18h] [rbp-78h] BYREF
  __int64 v39[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v40; // [rsp+30h] [rbp-60h]
  _QWORD v41[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v42; // [rsp+50h] [rbp-40h]

  v6 = *((unsigned int *)a2 + 2);
  v7 = *a2;
  if ( (_DWORD)v6 != 1 )
  {
    v8 = *(_QWORD *)(v7 + 8 * v6 - 8);
    v10 = v6 - 1;
    *((_DWORD *)a2 + 2) = v6 - 1;
    while ( 1 )
    {
      v13 = *(_BYTE *)(*(_QWORD *)v8 + 8LL);
      if ( v13 == 16 )
        v13 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v8 + 16LL) + 8LL);
      v14 = v10;
      v15 = v10 - 1;
      v16 = (__int64 *)(v7 + 8 * v14 - 8);
      v40 = 257;
      if ( v13 != 11 )
        break;
      v17 = *v16;
      *((_DWORD *)a2 + 2) = v15;
      if ( *(_BYTE *)(v8 + 16) > 0x10u || *(_BYTE *)(v17 + 16) > 0x10u )
      {
        v42 = 257;
        v29 = sub_15FB440(15, (__int64 *)v8, v17, (__int64)v41, 0);
        v30 = *(_QWORD *)(a1 + 8);
        v8 = v29;
        if ( v30 )
        {
          v31 = *(__int64 **)(a1 + 16);
          sub_157E9D0(v30 + 40, v29);
          v32 = *(_QWORD *)(v8 + 24);
          v33 = *v31;
          *(_QWORD *)(v8 + 32) = v31;
          v33 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v8 + 24) = v33 | v32 & 7;
          *(_QWORD *)(v33 + 8) = v8 + 24;
          *v31 = *v31 & 7 | (v8 + 24);
        }
        sub_164B780(v8, v39);
        v34 = *(_QWORD *)a1;
        if ( *(_QWORD *)a1 )
        {
          v38 = *(unsigned __int8 **)a1;
          sub_1623A60((__int64)&v38, v34, 2);
          v35 = *(_QWORD *)(v8 + 48);
          v36 = v8 + 48;
          if ( v35 )
          {
            sub_161E7C0(v8 + 48, v35);
            v36 = v8 + 48;
          }
          v37 = v38;
          *(_QWORD *)(v8 + 48) = v38;
          if ( v37 )
            sub_1623210((__int64)&v38, v37, v36);
        }
        goto LABEL_7;
      }
      v8 = sub_15A2C20((__int64 *)v8, v17, 0, 0, a3, a4, a5);
      v10 = *((_DWORD *)a2 + 2);
      if ( !v10 )
        return v8;
LABEL_8:
      v7 = *a2;
    }
    v11 = *v16;
    *((_DWORD *)a2 + 2) = v15;
    if ( *(_BYTE *)(v8 + 16) <= 0x10u
      && *(_BYTE *)(v11 + 16) <= 0x10u
      && (v12 = sub_15A2A30((__int64 *)0x10, (__int64 *)v8, v11, 0, 0, a3, a4, a5)) != 0 )
    {
      v8 = v12;
    }
    else
    {
      v42 = 257;
      v19 = sub_15FB440(16, (__int64 *)v8, v11, (__int64)v41, 0);
      v20 = *(_QWORD *)(a1 + 32);
      v21 = *(_DWORD *)(a1 + 40);
      v8 = v19;
      if ( v20 )
        sub_1625C10(v19, 3, v20);
      sub_15F2440(v8, v21);
      v22 = *(_QWORD *)(a1 + 8);
      if ( v22 )
      {
        v23 = *(__int64 **)(a1 + 16);
        sub_157E9D0(v22 + 40, v8);
        v24 = *(_QWORD *)(v8 + 24);
        v25 = *v23;
        *(_QWORD *)(v8 + 32) = v23;
        v25 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v8 + 24) = v25 | v24 & 7;
        *(_QWORD *)(v25 + 8) = v8 + 24;
        *v23 = *v23 & 7 | (v8 + 24);
      }
      sub_164B780(v8, v39);
      v26 = *(_QWORD *)a1;
      if ( *(_QWORD *)a1 )
      {
        v41[0] = *(_QWORD *)a1;
        sub_1623A60((__int64)v41, v26, 2);
        v27 = *(_QWORD *)(v8 + 48);
        if ( v27 )
          sub_161E7C0(v8 + 48, v27);
        v28 = (unsigned __int8 *)v41[0];
        *(_QWORD *)(v8 + 48) = v41[0];
        if ( v28 )
          sub_1623210((__int64)v41, v28, v8 + 48);
      }
    }
LABEL_7:
    v10 = *((_DWORD *)a2 + 2);
    if ( !v10 )
      return v8;
    goto LABEL_8;
  }
  return *(_QWORD *)v7;
}
