// Function: sub_24E9680
// Address: 0x24e9680
//
__int64 __fastcall sub_24E9680(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  int v6; // eax
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rcx
  _QWORD *v13; // rax
  __int64 v14; // r14
  unsigned int v15; // ebx
  __int64 v16; // r14
  unsigned __int8 *v17; // rbx
  __int64 v18; // rsi
  unsigned int **v19; // r13
  unsigned __int64 v20; // rsi
  __int64 v21; // r14
  __int64 v22; // rbx
  _QWORD *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // rax
  _QWORD *v27; // r15
  _QWORD *v28; // r12
  __int64 v29; // rax
  __int64 v30; // rbx
  __int64 v31; // rsi
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  __int64 v34; // [rsp+10h] [rbp-1C0h] BYREF
  __int64 v35; // [rsp+18h] [rbp-1B8h] BYREF
  __int64 v36[4]; // [rsp+20h] [rbp-1B0h] BYREF
  __int64 v37; // [rsp+40h] [rbp-190h]
  _BYTE *v38; // [rsp+48h] [rbp-188h]
  __int64 v39; // [rsp+50h] [rbp-180h]
  _BYTE v40[32]; // [rsp+58h] [rbp-178h] BYREF
  _BYTE *v41; // [rsp+78h] [rbp-158h]
  __int64 v42; // [rsp+80h] [rbp-150h]
  _BYTE v43[192]; // [rsp+88h] [rbp-148h] BYREF
  _BYTE *v44; // [rsp+148h] [rbp-88h]
  __int64 v45; // [rsp+150h] [rbp-80h]
  _BYTE v46[120]; // [rsp+158h] [rbp-78h] BYREF

  v5 = *(_QWORD *)(a1 + 24);
  v6 = *(_DWORD *)(v5 + 280);
  if ( v6 > 2 )
  {
    if ( v6 != 3 )
      goto LABEL_44;
    v10 = *(_QWORD *)(a1 + 296);
    v11 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
    v12 = *(_QWORD *)(v10 - 32 * v11);
    v13 = *(_QWORD **)(v12 + 24);
    if ( *(_DWORD *)(v12 + 32) > 0x40u )
      v13 = (_QWORD *)*v13;
    v14 = *(_QWORD *)(a1 + 280);
    v15 = (unsigned __int8)v13;
    if ( (*(_BYTE *)(v14 + 2) & 1) != 0 )
    {
      sub_B2C6D0(*(_QWORD *)(a1 + 280), a2, v11, v12);
      v11 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
    }
    v16 = *(_QWORD *)(v14 + 96) + 40LL * v15;
    v17 = sub_BD3990(*(unsigned __int8 **)(v10 + 32 * (2 - v11)), a2);
    v36[0] = *(_QWORD *)(a1 + 296);
    v18 = *(_QWORD *)(sub_24E84F0(a1 + 200, v36)[2] + 48LL);
    v34 = v18;
    if ( v18 )
      sub_B96E90((__int64)&v34, v18, 1);
    v19 = (unsigned int **)(a1 + 40);
    LOWORD(v37) = 257;
    v20 = *((_QWORD *)v17 + 3);
    v35 = v16;
    v21 = sub_921880((unsigned int **)(a1 + 40), v20, (int)v17, (int)&v35, 1, (__int64)v36, 0);
    *(_WORD *)(v21 + 2) = *(_WORD *)(v21 + 2) & 0xF003 | (4 * ((*((_WORD *)v17 + 1) >> 4) & 0x3FF));
    v36[0] = v34;
    if ( v34 )
    {
      v22 = v21 + 48;
      sub_B96E90((__int64)v36, v34, 1);
      if ( (__int64 *)(v21 + 48) == v36 )
      {
        if ( v36[0] )
          sub_B91220((__int64)v36, v36[0]);
        goto LABEL_19;
      }
      v32 = *(_QWORD *)(v21 + 48);
      if ( !v32 )
      {
LABEL_40:
        v33 = (unsigned __int8 *)v36[0];
        *(_QWORD *)(v21 + 48) = v36[0];
        if ( v33 )
          sub_B976B0((__int64)v36, v33, v22);
        goto LABEL_19;
      }
    }
    else
    {
      v22 = v21 + 48;
      if ( (__int64 *)(v21 + 48) == v36 || (v32 = *(_QWORD *)(v21 + 48)) == 0 )
      {
LABEL_19:
        v23 = *(_QWORD **)(a1 + 112);
        v36[0] = (__int64)"async.ctx.frameptr";
        v24 = *(_QWORD *)(a1 + 24);
        LOWORD(v37) = 259;
        v25 = *(_QWORD *)(v24 + 360);
        v26 = sub_BCB2B0(v23);
        v8 = sub_94B2B0(v19, v26, v21, v25, (__int64)v36);
        v46[64] = 1;
        v39 = 0x400000000LL;
        v41 = v43;
        memset(v36, 0, sizeof(v36));
        v37 = 0;
        v38 = v40;
        v42 = 0x800000000LL;
        v44 = v46;
        v45 = 0x800000000LL;
        sub_29F2700(v21, v36, 0, 0, 1, 0);
        if ( v44 != v46 )
          _libc_free((unsigned __int64)v44);
        v27 = v41;
        v28 = &v41[24 * (unsigned int)v42];
        if ( v41 != (_BYTE *)v28 )
        {
          do
          {
            v29 = *(v28 - 1);
            v28 -= 3;
            if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
              sub_BD60C0(v28);
          }
          while ( v27 != v28 );
          v28 = v41;
        }
        if ( v28 != (_QWORD *)v43 )
          _libc_free((unsigned __int64)v28);
        if ( v38 != v40 )
          _libc_free((unsigned __int64)v38);
        if ( v34 )
          sub_B91220((__int64)&v34, v34);
        return v8;
      }
    }
    sub_B91220(v22, v32);
    goto LABEL_40;
  }
  if ( v6 <= 0 )
  {
    if ( !v6 )
    {
      v7 = *(_QWORD *)(a1 + 280);
      if ( (*(_BYTE *)(v7 + 2) & 1) != 0 )
        sub_B2C6D0(*(_QWORD *)(a1 + 280), a2, v5, a4);
      return *(_QWORD *)(v7 + 96);
    }
LABEL_44:
    BUG();
  }
  v30 = *(_QWORD *)(a1 + 280);
  if ( (*(_BYTE *)(v30 + 2) & 1) != 0 )
  {
    sub_B2C6D0(*(_QWORD *)(a1 + 280), a2, v5, a4);
    v5 = *(_QWORD *)(a1 + 24);
  }
  v8 = *(_QWORD *)(v30 + 96);
  v31 = sub_BCE3C0(**(__int64 ***)(v5 + 288), 0);
  if ( !*(_BYTE *)(*(_QWORD *)(a1 + 24) + 360LL) )
  {
    LOWORD(v37) = 257;
    return sub_A82CA0((unsigned int **)(a1 + 40), v31, v8, 0, 0, (__int64)v36);
  }
  return v8;
}
