// Function: sub_2475030
// Address: 0x2475030
//
void __fastcall sub_2475030(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v8; // rsi
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 **v15; // rcx
  __int64 v16; // rax
  int v17; // esi
  unsigned __int64 v18; // rax
  __int64 **v19; // rcx
  __int64 v20; // rax
  int v21; // esi
  _BYTE *v22; // rax
  _BYTE *v23; // rax
  __int64 v24; // [rsp+8h] [rbp-168h]
  __int64 v25; // [rsp+10h] [rbp-160h]
  __int64 v26; // [rsp+18h] [rbp-158h]
  __int64 v27; // [rsp+20h] [rbp-150h]
  _BYTE *v28; // [rsp+28h] [rbp-148h]
  _BYTE *v29; // [rsp+30h] [rbp-140h]
  __int64 v30; // [rsp+38h] [rbp-138h]
  unsigned __int64 v31; // [rsp+40h] [rbp-130h]
  int v33[8]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v34; // [rsp+70h] [rbp-100h]
  _QWORD v35[4]; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v36; // [rsp+A0h] [rbp-D0h]
  unsigned int *v37[2]; // [rsp+B0h] [rbp-C0h] BYREF
  char v38; // [rsp+C0h] [rbp-B0h] BYREF
  void *v39; // [rsp+130h] [rbp-40h]

  v31 = a4;
  sub_23D0AB0((__int64)v37, a2, 0, 0, 0);
  v30 = sub_246F3F0((__int64)a1, a3);
  v29 = (_BYTE *)sub_246F3F0((__int64)a1, a4);
  v26 = 0;
  v28 = (_BYTE *)sub_246F3F0((__int64)a1, a5);
  if ( *(_DWORD *)(a1[1] + 4) && (v26 = sub_246EE10((__int64)a1, a3), *(_DWORD *)(a1[1] + 4)) )
  {
    v24 = sub_246EE10((__int64)a1, v31);
    if ( *(_DWORD *)(a1[1] + 4) )
      v25 = sub_246EE10((__int64)a1, a5);
    else
      v25 = 0;
  }
  else
  {
    v24 = 0;
    v25 = 0;
  }
  LOWORD(v36) = 257;
  v27 = sub_B36550(v37, a3, (__int64)v29, (__int64)v28, (__int64)v35, 0);
  v8 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 15 > 1 )
  {
    v15 = (__int64 **)sub_2463540(a1, *(_QWORD *)(v31 + 8));
    v16 = *(_QWORD *)(v31 + 8);
    if ( v15 != (__int64 **)v16 )
    {
      v17 = *(unsigned __int8 *)(v16 + 8);
      if ( (unsigned int)(v17 - 17) <= 1 )
        LOBYTE(v17) = *(_BYTE *)(**(_QWORD **)(v16 + 16) + 8LL);
      LOWORD(v36) = 257;
      if ( (_BYTE)v17 == 14 )
        v18 = sub_24633A0((__int64 *)v37, 0x2Fu, v31, v15, (__int64)v35, 0, v33[0], 0);
      else
        v18 = sub_24633A0((__int64 *)v37, 0x31u, v31, v15, (__int64)v35, 0, v33[0], 0);
      v31 = v18;
    }
    v19 = (__int64 **)sub_2463540(a1, *(_QWORD *)(a5 + 8));
    v20 = *(_QWORD *)(a5 + 8);
    if ( v19 != (__int64 **)v20 )
    {
      v21 = *(unsigned __int8 *)(v20 + 8);
      if ( (unsigned int)(v21 - 17) <= 1 )
        LOBYTE(v21) = *(_BYTE *)(**(_QWORD **)(v20 + 16) + 8LL);
      LOWORD(v36) = 257;
      if ( (_BYTE)v21 == 14 )
        a5 = sub_24633A0((__int64 *)v37, 0x2Fu, a5, v19, (__int64)v35, 0, v33[0], 0);
      else
        a5 = sub_24633A0((__int64 *)v37, 0x31u, a5, v19, (__int64)v35, 0, v33[0], 0);
    }
    v34 = 257;
    v22 = (_BYTE *)sub_A825B0(v37, (_BYTE *)v31, (_BYTE *)a5, (__int64)v33);
    LOWORD(v36) = 257;
    v23 = (_BYTE *)sub_A82480(v37, v22, v29, (__int64)v35);
    v36 = 257;
    v10 = sub_A82480(v37, v23, v28, (__int64)v35);
  }
  else
  {
    v9 = sub_2463540(a1, v8);
    v10 = sub_24623D0((__int64)v9);
  }
  v35[0] = "_msprop_select";
  LOWORD(v36) = 259;
  v11 = sub_B36550(v37, v30, v10, v27, (__int64)v35, 0);
  sub_246EF60((__int64)a1, a2, v11);
  if ( *(_DWORD *)(a1[1] + 4) )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a3 + 8) + 8LL) - 17 <= 1 )
    {
      LOWORD(v36) = 257;
      v12 = sub_2465600((__int64)a1, a3, (__int64)v37, (__int64)v35);
      LOWORD(v36) = 257;
      a3 = v12;
      v30 = sub_2465600((__int64)a1, v30, (__int64)v37, (__int64)v35);
    }
    v34 = 257;
    LOWORD(v36) = 257;
    v13 = sub_B36550(v37, a3, v24, v25, (__int64)v33, 0);
    v14 = sub_B36550(v37, v30, v26, v13, (__int64)v35, 0);
    sub_246F1C0((__int64)a1, a2, v14);
  }
  nullsub_61();
  v39 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v37[0] != &v38 )
    _libc_free((unsigned __int64)v37[0]);
}
