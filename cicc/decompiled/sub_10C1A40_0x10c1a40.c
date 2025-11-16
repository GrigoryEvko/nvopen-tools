// Function: sub_10C1A40
// Address: 0x10c1a40
//
__int64 __fastcall sub_10C1A40(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3, char a4)
{
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rdx
  int v8; // ecx
  int v9; // esi
  __int64 result; // rax
  int v11; // eax
  __int64 *v12; // rsi
  __int64 v13; // r12
  _BYTE *v14; // rax
  unsigned int **v15; // rdi
  int v16; // ecx
  int v17; // eax
  int v18; // [rsp+Ch] [rbp-134h]
  int v19; // [rsp+10h] [rbp-130h]
  int v20; // [rsp+14h] [rbp-12Ch]
  int v21; // [rsp+18h] [rbp-128h]
  int v22; // [rsp+1Ch] [rbp-124h]
  __int64 v23; // [rsp+20h] [rbp-120h]
  int v24; // [rsp+28h] [rbp-118h]
  char v25; // [rsp+2Fh] [rbp-111h]
  unsigned int v26; // [rsp+34h] [rbp-10Ch] BYREF
  unsigned int *v27; // [rsp+38h] [rbp-108h] BYREF
  __int64 v28; // [rsp+40h] [rbp-100h] BYREF
  int v29; // [rsp+48h] [rbp-F8h]
  int v30; // [rsp+4Ch] [rbp-F4h]
  __int64 v31; // [rsp+50h] [rbp-F0h] BYREF
  int v32; // [rsp+58h] [rbp-E8h]
  int v33; // [rsp+5Ch] [rbp-E4h]
  __int64 v34; // [rsp+60h] [rbp-E0h] BYREF
  int v35; // [rsp+68h] [rbp-D8h]
  int v36; // [rsp+6Ch] [rbp-D4h]
  char v37; // [rsp+70h] [rbp-D0h]
  __int64 v38; // [rsp+80h] [rbp-C0h] BYREF
  int v39; // [rsp+88h] [rbp-B8h]
  int v40; // [rsp+8Ch] [rbp-B4h]
  char v41; // [rsp+90h] [rbp-B0h]
  __int64 v42; // [rsp+A0h] [rbp-A0h] BYREF
  int v43; // [rsp+A8h] [rbp-98h]
  int v44; // [rsp+ACh] [rbp-94h]
  char v45; // [rsp+B0h] [rbp-90h]
  __int64 v46; // [rsp+C0h] [rbp-80h] BYREF
  int v47; // [rsp+C8h] [rbp-78h]
  int v48; // [rsp+CCh] [rbp-74h]
  char v49; // [rsp+D0h] [rbp-70h]
  _BYTE v50[32]; // [rsp+E0h] [rbp-60h] BYREF
  __int16 v51; // [rsp+100h] [rbp-40h]

  v26 = (a4 == 0) + 32;
  v27 = &v26;
  sub_10C1460((__int64)&v34, &v27, a2, 0);
  v5 = v34;
  v24 = v35;
  v20 = v36;
  sub_10C1460((__int64)&v38, &v27, a2, 1u);
  v23 = v38;
  v19 = v39;
  v18 = v40;
  sub_10C1460((__int64)&v42, &v27, a3, 0);
  v6 = v42;
  v22 = v43;
  v21 = v44;
  v25 = v45;
  sub_10C1460((__int64)&v46, &v27, a3, 1u);
  if ( !v37 || !v41 || !v25 || !v49 )
    return 0;
  v7 = v46;
  v8 = v47;
  v9 = v48;
  if ( v6 == v5 && v23 == v46 )
  {
    v9 = v21;
    v7 = v6;
    v6 = v23;
    v21 = v48;
    v8 = v22;
    v22 = v47;
  }
  else
  {
    result = 0;
    if ( v46 != v5 || v23 != v6 )
      return result;
  }
  if ( v24 + v20 == v8 && v18 + v19 == v22 )
  {
    v16 = v18;
    v7 = v5;
    v6 = v23;
    v18 = v21;
    v21 = v16;
    v8 = v24;
    v22 = v19;
    v17 = v9;
    v9 = v20;
    v20 = v17;
LABEL_11:
    v28 = v7;
    v29 = v8;
    v11 = v9 + v20;
    v12 = *(__int64 **)(a1 + 32);
    v31 = v6;
    v30 = v11;
    v32 = v22;
    v33 = v21 + v18;
    v13 = sub_10B9020((__int64)&v28, v12);
    v14 = (_BYTE *)sub_10B9020((__int64)&v31, *(__int64 **)(a1 + 32));
    v15 = *(unsigned int ***)(a1 + 32);
    v51 = 257;
    return sub_92B530(v15, v26, v13, v14, (__int64)v50);
  }
  result = 0;
  if ( v9 + v8 == v24 && v21 + v22 == v19 )
    goto LABEL_11;
  return result;
}
