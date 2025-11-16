// Function: sub_2900420
// Address: 0x2900420
//
void __fastcall sub_2900420(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // r14d
  _QWORD **v7; // r13
  __int64 *v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r13
  __int16 v13; // dx
  __int64 v14; // r8
  _QWORD *v15; // rax
  __int64 v16; // r9
  __int64 v17; // r8
  unsigned __int16 v18; // dx
  __int64 v19; // rdx
  __int16 v20; // dx
  __int64 v21; // r8
  char v22; // r12
  _QWORD *v23; // rax
  __int64 v24; // r9
  __int64 v25; // r8
  unsigned __int16 v26; // dx
  __int64 v27; // rdx
  __int64 v28; // r8
  _QWORD *v29; // rax
  __int64 v30; // r9
  __int64 v31; // r12
  __int64 v32; // rdx
  __int64 v33; // [rsp-10h] [rbp-A0h]
  char v34; // [rsp+8h] [rbp-88h]
  char v35; // [rsp+10h] [rbp-80h]
  char v36; // [rsp+10h] [rbp-80h]
  __int64 v37; // [rsp+18h] [rbp-78h]
  __int64 v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+18h] [rbp-78h]
  __int64 v40; // [rsp+18h] [rbp-78h]
  __int64 v41; // [rsp+18h] [rbp-78h]
  _QWORD *v42; // [rsp+18h] [rbp-78h]
  __int64 v43; // [rsp+20h] [rbp-70h]
  _QWORD *v44; // [rsp+28h] [rbp-68h]
  _BYTE v45[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v46; // [rsp+50h] [rbp-40h]

  if ( a3 )
  {
    v6 = a3 + 1;
    v7 = (_QWORD **)sub_B43CA0(a1);
    v8 = (__int64 *)sub_BCB120(*v7);
    v9 = sub_BCF640(v8, 1u);
    v10 = sub_BA8CA0((__int64)v7, (__int64)"__tmp_use", 9u, v9);
    v43 = v11;
    v12 = v10;
    if ( *(_BYTE *)a1 == 85 )
    {
      v28 = *(_QWORD *)(a1 + 32);
      v46 = 257;
      v41 = v28;
      v29 = sub_BD2C40(88, v6);
      v31 = (__int64)v29;
      if ( v29 )
      {
        sub_B44260((__int64)v29, **(_QWORD **)(v12 + 16), 56, v6 & 0x7FFFFFF, v41, 0);
        *(_QWORD *)(v31 + 72) = 0;
        sub_B4A290(v31, v12, v43, a2, a3, (__int64)v45, 0, 0);
        v30 = v33;
      }
      v32 = *(unsigned int *)(a4 + 8);
      if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        sub_C8D5F0(a4, (const void *)(a4 + 16), v32 + 1, 8u, v32 + 1, v30);
        v32 = *(unsigned int *)(a4 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v32) = v31;
      ++*(_DWORD *)(a4 + 8);
    }
    else
    {
      v14 = sub_AA5190(*(_QWORD *)(a1 - 96));
      if ( v14 )
      {
        v35 = v13;
        v34 = HIBYTE(v13);
      }
      else
      {
        v34 = 0;
        v35 = 0;
      }
      v37 = v14;
      v46 = 257;
      v15 = sub_BD2C40(88, v6);
      if ( v15 )
      {
        v17 = v37;
        v38 = (__int64)v15;
        LOBYTE(v18) = v35;
        HIBYTE(v18) = v34;
        sub_B44260((__int64)v15, **(_QWORD **)(v12 + 16), 56, v6 & 0x7FFFFFF, v17, v18);
        *(_QWORD *)(v38 + 72) = 0;
        sub_B4A290(v38, v12, v43, a2, a3, (__int64)v45, 0, 0);
        v15 = (_QWORD *)v38;
      }
      v19 = *(unsigned int *)(a4 + 8);
      if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        v42 = v15;
        sub_C8D5F0(a4, (const void *)(a4 + 16), v19 + 1, 8u, v19 + 1, v16);
        v19 = *(unsigned int *)(a4 + 8);
        v15 = v42;
      }
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v19) = v15;
      ++*(_DWORD *)(a4 + 8);
      v21 = sub_AA5190(*(_QWORD *)(a1 - 64));
      if ( v21 )
      {
        v22 = v20;
        v36 = HIBYTE(v20);
      }
      else
      {
        v36 = 0;
        v22 = 0;
      }
      v39 = v21;
      v46 = 257;
      v23 = sub_BD2C40(88, v6);
      if ( v23 )
      {
        v25 = v39;
        v40 = (__int64)v23;
        LOBYTE(v26) = v22;
        HIBYTE(v26) = v36;
        sub_B44260((__int64)v23, **(_QWORD **)(v12 + 16), 56, v6 & 0x7FFFFFF, v25, v26);
        *(_QWORD *)(v40 + 72) = 0;
        sub_B4A290(v40, v12, v43, a2, a3, (__int64)v45, 0, 0);
        v23 = (_QWORD *)v40;
      }
      v27 = *(unsigned int *)(a4 + 8);
      if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        v44 = v23;
        sub_C8D5F0(a4, (const void *)(a4 + 16), v27 + 1, 8u, v27 + 1, v24);
        v27 = *(unsigned int *)(a4 + 8);
        v23 = v44;
      }
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v27) = v23;
      ++*(_DWORD *)(a4 + 8);
    }
  }
}
