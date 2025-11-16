// Function: sub_11E4D30
// Address: 0x11e4d30
//
_BYTE *__fastcall sub_11E4D30(_QWORD *a1, __int64 a2, __int64 a3)
{
  char v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int8 *v10; // r14
  unsigned __int8 *v11; // r10
  __int64 v12; // rdi
  int v13; // ebx
  __int64 (__fastcall *v14)(__int64, __int64, unsigned __int8 *, unsigned __int8 *, __int64); // r9
  unsigned int v15; // eax
  __int64 v16; // r8
  __int64 v17; // rax
  _BYTE *v18; // r15
  int v19; // eax
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // r8
  __int64 *v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rbx
  __int64 v32; // r12
  __int64 v33; // rdx
  unsigned int v34; // esi
  __int64 v35; // rax
  unsigned __int8 *v36; // [rsp+8h] [rbp-C8h]
  unsigned __int8 *v37; // [rsp+8h] [rbp-C8h]
  __int64 v38; // [rsp+10h] [rbp-C0h]
  _QWORD v39[4]; // [rsp+20h] [rbp-B0h] BYREF
  __int16 v40; // [rsp+40h] [rbp-90h]
  __m128i v41; // [rsp+50h] [rbp-80h] BYREF
  __int64 v42; // [rsp+60h] [rbp-70h]
  __int64 v43; // [rsp+68h] [rbp-68h]
  __int64 v44; // [rsp+70h] [rbp-60h]
  __int64 v45; // [rsp+78h] [rbp-58h]
  __int64 v46; // [rsp+80h] [rbp-50h]
  __int64 v47; // [rsp+88h] [rbp-48h]
  __int16 v48; // [rsp+90h] [rbp-40h]

  if ( sub_B451C0(a2) )
  {
    v6 = 0;
    v40 = 257;
    if ( !a2 )
      goto LABEL_4;
  }
  else
  {
    v19 = *(_DWORD *)(a2 + 4);
    v20 = a1[4];
    v45 = a2;
    v21 = a1[6];
    v22 = a1[3];
    v42 = 0;
    v23 = a1[2];
    v43 = v20;
    v41.m128i_i64[1] = v22;
    v44 = v21;
    v24 = *(__int64 **)(a2 - 32LL * (v19 & 0x7FFFFFF));
    v25 = a1[5];
    v48 = 257;
    v41.m128i_i64[0] = v23;
    v46 = v25;
    v47 = 0;
    if ( (sub_9B4030(v24, 516, 0, &v41) & 0x204) != 0 )
      return 0;
    v26 = sub_9B4030(*(__int64 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), 240, 0, &v41);
    v27 = *(_QWORD *)(a2 + 8);
    v39[0] = v26;
    if ( !sub_989140(v39, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL), v27) )
      return 0;
    v40 = 257;
  }
  v6 = 1;
  LODWORD(v38) = sub_B45210(a2);
LABEL_4:
  BYTE4(v38) = v6;
  v7 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v8 = 32 * (1 - v7);
  v9 = -32 * v7;
  v10 = *(unsigned __int8 **)(a2 + v8);
  v11 = *(unsigned __int8 **)(a2 + v9);
  if ( !*(_BYTE *)(a3 + 108) )
  {
    v12 = *(_QWORD *)(a3 + 80);
    v13 = v38;
    v14 = *(__int64 (__fastcall **)(__int64, __int64, unsigned __int8 *, unsigned __int8 *, __int64))(*(_QWORD *)v12 + 40LL);
    v15 = *(_DWORD *)(a3 + 104);
    v16 = v15;
    if ( v6 )
      v16 = (unsigned int)v38;
    if ( (char *)v14 == (char *)sub_928A40 )
    {
      if ( *v11 > 0x15u || *v10 > 0x15u )
        goto LABEL_21;
      v36 = v11;
      if ( (unsigned __int8)sub_AC47B0(24) )
        v17 = sub_AD5570(24, (__int64)v36, v10, 0, 0);
      else
        v17 = sub_AABE40(0x18u, v36, v10);
      v11 = v36;
      v18 = (_BYTE *)v17;
    }
    else
    {
      v37 = v11;
      v35 = v14(v12, 24, v11, v10, v16);
      v11 = v37;
      v18 = (_BYTE *)v35;
    }
    if ( v18 )
      goto LABEL_14;
    v15 = *(_DWORD *)(a3 + 104);
LABEL_21:
    if ( !v6 )
      v13 = v15;
    LOWORD(v44) = 257;
    v29 = sub_B504D0(24, (__int64)v11, (__int64)v10, (__int64)&v41, 0, 0);
    v30 = *(_QWORD *)(a3 + 96);
    v18 = (_BYTE *)v29;
    if ( v30 )
      sub_B99FD0(v29, 3u, v30);
    sub_B45150((__int64)v18, v13);
    (*(void (__fastcall **)(_QWORD, _BYTE *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
      *(_QWORD *)(a3 + 88),
      v18,
      v39,
      *(_QWORD *)(a3 + 56),
      *(_QWORD *)(a3 + 64));
    v31 = *(_QWORD *)a3;
    v32 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
    if ( *(_QWORD *)a3 != v32 )
    {
      do
      {
        v33 = *(_QWORD *)(v31 + 8);
        v34 = *(_DWORD *)v31;
        v31 += 16;
        sub_B99FD0((__int64)v18, v34, v33);
      }
      while ( v32 != v31 );
    }
    goto LABEL_14;
  }
  v18 = (_BYTE *)sub_B35400(a3, 0x72u, *(_QWORD *)(a2 + v9), *(_QWORD *)(a2 + v8), v38, (__int64)v39, 0, 0, 0);
LABEL_14:
  if ( *v18 > 0x1Cu )
    sub_B44EF0((__int64)v18, 1);
  return v18;
}
