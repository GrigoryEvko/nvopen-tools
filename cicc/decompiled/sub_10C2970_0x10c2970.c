// Function: sub_10C2970
// Address: 0x10c2970
//
__int64 __fastcall sub_10C2970(__int64 a1, __int64 a2, char a3, const __m128i *a4, __int64 a5)
{
  _BYTE *v7; // rdx
  _BYTE *v8; // r10
  int v9; // ecx
  __int64 v10; // r10
  __int64 v11; // rdx
  __int64 v12; // r11
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // r11
  _BYTE *v19; // r10
  __int64 v20; // r13
  __int64 v21; // rax
  _BYTE *v22; // r10
  __int64 v23; // r14
  char v24; // al
  __int64 v25; // r11
  _BYTE *v26; // r10
  unsigned __int8 *v27; // rax
  __int64 v28; // rbx
  __int64 v29; // r13
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // [rsp+0h] [rbp-E0h]
  __int64 v33; // [rsp+0h] [rbp-E0h]
  __int64 v34; // [rsp+0h] [rbp-E0h]
  _BYTE *v35; // [rsp+0h] [rbp-E0h]
  __int64 v36; // [rsp+0h] [rbp-E0h]
  _BYTE *v37; // [rsp+0h] [rbp-E0h]
  __int64 v39; // [rsp+8h] [rbp-D8h]
  __int64 v40; // [rsp+8h] [rbp-D8h]
  __int64 v41; // [rsp+8h] [rbp-D8h]
  __int64 v42; // [rsp+8h] [rbp-D8h]
  _BYTE *v43; // [rsp+8h] [rbp-D8h]
  _BYTE *v44; // [rsp+10h] [rbp-D0h] BYREF
  int v45; // [rsp+18h] [rbp-C8h] BYREF
  char v46; // [rsp+1Ch] [rbp-C4h]
  char v47[32]; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v48; // [rsp+40h] [rbp-A0h]
  _BYTE v49[32]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v50; // [rsp+70h] [rbp-70h]
  __int64 v51[4]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v52; // [rsp+A0h] [rbp-40h]

  v51[0] = (__int64)&v45;
  v45 = 42;
  v46 = 0;
  v51[1] = (__int64)&v44;
  if ( !(unsigned __int8)sub_10C27C0(v51, a1) || (unsigned int)(v45 - 32) > 1 || !a2 )
    return 0;
  v7 = *(_BYTE **)(a2 - 64);
  v8 = *(_BYTE **)(a2 - 32);
  if ( v44 == v7 && v8 )
  {
    v33 = *(_QWORD *)(a2 - 32);
    v14 = sub_B53900(a2);
    v10 = v33;
    v9 = v14;
  }
  else
  {
    v32 = *(_QWORD *)(a2 - 64);
    if ( !v7 || v44 != v8 )
      return 0;
    v9 = sub_B53960(a2);
    v10 = v32;
  }
  if ( *v44 != 42 )
    return 0;
  v11 = *((_QWORD *)v44 - 8);
  v12 = *((_QWORD *)v44 - 4);
  if ( v11 == v10 )
  {
    if ( v12 )
      goto LABEL_16;
    return 0;
  }
  if ( v12 != v10 || !v11 )
    return 0;
  v12 = *((_QWORD *)v44 - 8);
LABEL_16:
  v15 = *(_QWORD *)(a1 + 16);
  if ( !v15 || *(_QWORD *)(v15 + 8) )
  {
    v16 = *(_QWORD *)(a2 + 16);
    if ( !v16 || *(_QWORD *)(v16 + 8) )
      return 0;
  }
  if ( v9 == 36 )
  {
    if ( v45 != 33 || !a3 )
      return 0;
    v36 = v10;
    v41 = v12;
    v24 = sub_9B6260(v12, a4, 0);
    v25 = v41;
    v26 = (_BYTE *)v36;
    if ( !v24 )
    {
      v25 = v36;
      v26 = (_BYTE *)v41;
    }
    v37 = v26;
    v42 = v25;
    if ( !(unsigned __int8)sub_9B6260(v25, a4, 0) )
      return 0;
    v52 = 257;
    v50 = 257;
    v27 = sub_10A0530((__int64 *)a5, v42, (__int64)v49, 0);
    return sub_92B530((unsigned int **)a5, 0x24u, (__int64)v27, v37, (__int64)v51);
  }
  else
  {
    if ( v9 != 35 )
      return 0;
    v34 = v10;
    if ( v45 != 32 || a3 )
      return 0;
    v39 = v12;
    v17 = sub_9B6260(v12, a4, 0);
    v18 = v39;
    v19 = (_BYTE *)v34;
    if ( !v17 )
    {
      v19 = (_BYTE *)v39;
      v18 = v34;
    }
    v35 = v19;
    v40 = v18;
    if ( !(unsigned __int8)sub_9B6260(v18, a4, 0) )
      return 0;
    v48 = 257;
    v50 = 257;
    v20 = sub_AD6530(*(_QWORD *)(v40 + 8), (__int64)a4);
    v21 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a5 + 80) + 32LL))(
            *(_QWORD *)(a5 + 80),
            15,
            v20,
            v40,
            0,
            0);
    v22 = v35;
    v23 = v21;
    if ( !v21 )
    {
      v52 = 257;
      v23 = sub_B504D0(15, v20, v40, (__int64)v51, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
        *(_QWORD *)(a5 + 88),
        v23,
        v47,
        *(_QWORD *)(a5 + 56),
        *(_QWORD *)(a5 + 64));
      v28 = *(_QWORD *)a5;
      v22 = v35;
      v29 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
      if ( *(_QWORD *)a5 != v29 )
      {
        do
        {
          v30 = *(_QWORD *)(v28 + 8);
          v31 = *(_DWORD *)v28;
          v28 += 16;
          v43 = v22;
          sub_B99FD0(v23, v31, v30);
          v22 = v43;
        }
        while ( v29 != v28 );
      }
    }
    return sub_92B530((unsigned int **)a5, 0x23u, v23, v22, (__int64)v49);
  }
}
