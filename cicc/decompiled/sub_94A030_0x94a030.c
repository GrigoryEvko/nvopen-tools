// Function: sub_94A030
// Address: 0x94a030
//
__int64 __fastcall sub_94A030(__int64 a1, __int64 a2, int a3, __int64 a4, char a5)
{
  int v5; // r15d
  __m128i *v9; // r14
  unsigned int v10; // r15d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v15; // rax
  int v16; // r15d
  __int64 v17; // rax
  int v18; // r9d
  __int64 v19; // rbx
  __int64 v20; // rdx
  unsigned int *v21; // r15
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // r14
  __int64 v26; // rax
  unsigned __int8 v27; // al
  int v28; // ebx
  __int64 v29; // rax
  __int64 v30; // r9
  __int64 v31; // r15
  __int64 v32; // rdi
  unsigned int *v33; // r14
  unsigned int *v34; // rbx
  __int64 v35; // rdx
  __int64 v36; // rsi
  __int64 (__fastcall *v37)(__int64, __int64, __int64); // rax
  unsigned int *v38; // r14
  unsigned int *v39; // rbx
  __int64 v40; // rdx
  __int64 v41; // rsi
  unsigned int v42; // eax
  __int64 v43; // rax
  __int64 v44; // rax
  unsigned int *v45; // r14
  unsigned int *v46; // rbx
  __int64 v47; // rdx
  __int64 v48; // rsi
  __int64 v49; // [rsp-10h] [rbp-D0h]
  int v50; // [rsp+8h] [rbp-B8h]
  __int64 v51; // [rsp+10h] [rbp-B0h]
  __int64 v52; // [rsp+20h] [rbp-A0h]
  unsigned int *v54; // [rsp+28h] [rbp-98h]
  unsigned int v55[8]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v56; // [rsp+50h] [rbp-70h]
  _QWORD v57[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v58; // [rsp+80h] [rbp-40h]

  v5 = a3 - 194;
  v9 = sub_92F410(a2, *(_QWORD *)(*(_QWORD *)(a4 + 72) + 16LL));
  switch ( v5 )
  {
    case 0:
    case 4:
      v10 = 3;
      break;
    case 1:
    case 5:
      v10 = 4;
      break;
    case 2:
    case 6:
      v10 = 5;
      break;
    default:
      v10 = 1;
      break;
  }
  v11 = sub_BCB2B0(*(_QWORD *)(a2 + 40));
  v51 = sub_BCE760(v11, v10);
  v52 = a2 + 48;
  if ( a5 )
  {
    v58 = 257;
    v12 = sub_949E90((unsigned int **)(a2 + 48), 0x30u, (__int64)v9, v51, (__int64)v57, 0, v55[0], 0);
    v13 = sub_92CAE0(a2, v12, a4 + 36);
  }
  else
  {
    v57[0] = "temp";
    v58 = 259;
    v50 = sub_921B80(a2, v9->m128i_i64[1], (__int64)v57, 0, 0);
    v15 = sub_AA4E30(*(_QWORD *)(a2 + 96));
    v16 = (unsigned __int8)sub_AE5020(v15, v9->m128i_i64[1]);
    v58 = 257;
    v17 = sub_BD2C40(80, unk_3F10A10);
    v19 = v17;
    if ( v17 )
      sub_B4D3C0(v17, (_DWORD)v9, v50, 0, v16, v18, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
      *(_QWORD *)(a2 + 136),
      v19,
      v57,
      *(_QWORD *)(a2 + 104),
      *(_QWORD *)(a2 + 112));
    v20 = 4LL * *(unsigned int *)(a2 + 56);
    v21 = *(unsigned int **)(a2 + 48);
    v54 = &v21[v20];
    while ( v54 != v21 )
    {
      v22 = *((_QWORD *)v21 + 1);
      v23 = *v21;
      v21 += 4;
      sub_B99FD0(v19, v23, v22);
    }
    v24 = *(_QWORD *)(a2 + 96);
    v56 = 257;
    v25 = v9->m128i_i64[1];
    v26 = sub_AA4E30(v24);
    v27 = sub_AE5020(v26, v25);
    v58 = 257;
    v28 = v27;
    v29 = sub_BD2C40(80, unk_3F10A14);
    v31 = v29;
    if ( v29 )
    {
      sub_B4D190(v29, v25, v50, (unsigned int)v57, 0, v28, 0, 0);
      v30 = v49;
    }
    v32 = *(_QWORD *)(a2 + 136);
    (*(void (__fastcall **)(__int64, __int64, unsigned int *, _QWORD, _QWORD, __int64))(*(_QWORD *)v32 + 16LL))(
      v32,
      v31,
      v55,
      *(_QWORD *)(v52 + 56),
      *(_QWORD *)(v52 + 64),
      v30);
    v33 = *(unsigned int **)(a2 + 48);
    v34 = &v33[4 * *(unsigned int *)(a2 + 56)];
    while ( v34 != v33 )
    {
      v35 = *((_QWORD *)v33 + 1);
      v36 = *v33;
      v32 = v31;
      v33 += 4;
      sub_B99FD0(v31, v36, v35);
    }
    v56 = 257;
    if ( v51 != *(_QWORD *)(v31 + 8) )
    {
      if ( *(_BYTE *)v31 > 0x15u )
      {
        v58 = 257;
        v44 = sub_B52210(v31, v51, v57, 0, 0);
        v32 = *(_QWORD *)(a2 + 136);
        v31 = v44;
        (*(void (__fastcall **)(__int64, __int64, unsigned int *, _QWORD, _QWORD))(*(_QWORD *)v32 + 16LL))(
          v32,
          v44,
          v55,
          *(_QWORD *)(v52 + 56),
          *(_QWORD *)(v52 + 64));
        v45 = *(unsigned int **)(a2 + 48);
        v46 = &v45[4 * *(unsigned int *)(a2 + 56)];
        while ( v46 != v45 )
        {
          v47 = *((_QWORD *)v45 + 1);
          v48 = *v45;
          v32 = v31;
          v45 += 4;
          sub_B99FD0(v31, v48, v47);
        }
      }
      else
      {
        v32 = *(_QWORD *)(a2 + 128);
        v37 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v32 + 136LL);
        if ( v37 == sub_928970 )
        {
          v32 = v31;
          v31 = sub_ADAFB0(v31, v51);
        }
        else
        {
          v31 = v37(v32, v31, v51);
        }
        if ( *(_BYTE *)v31 > 0x1Cu )
        {
          v32 = *(_QWORD *)(a2 + 136);
          (*(void (__fastcall **)(__int64, __int64, unsigned int *, _QWORD, _QWORD))(*(_QWORD *)v32 + 16LL))(
            v32,
            v31,
            v55,
            *(_QWORD *)(v52 + 56),
            *(_QWORD *)(v52 + 64));
          v38 = *(unsigned int **)(a2 + 48);
          v39 = &v38[4 * *(unsigned int *)(a2 + 56)];
          while ( v39 != v38 )
          {
            v40 = *((_QWORD *)v38 + 1);
            v41 = *v38;
            v32 = v31;
            v38 += 4;
            sub_B99FD0(v31, v41, v40);
          }
        }
      }
    }
    v42 = sub_91B6E0(v32);
    v43 = sub_BCCE00(*(_QWORD *)(a2 + 40), v42);
    v58 = 257;
    v13 = sub_949E90((unsigned int **)v52, 0x2Fu, v31, v43, (__int64)v57, 0, v55[0], 0);
  }
  *(_QWORD *)a1 = v13;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
