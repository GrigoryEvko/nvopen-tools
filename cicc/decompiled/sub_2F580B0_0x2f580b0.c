// Function: sub_2F580B0
// Address: 0x2f580b0
//
__int64 __fastcall sub_2F580B0(__int64 a1, unsigned __int16 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r12
  int v9; // ecx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  unsigned __int64 v13; // r12
  __int64 v14; // rsi
  __int64 (__fastcall *v15)(__int64); // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r9
  _DWORD *v19; // rax
  _DWORD *v20; // rdx
  __int64 v21; // rax
  int v22; // r13d
  unsigned __int64 v23; // rsi
  __int64 v24; // rdi
  unsigned __int64 i; // rax
  __int64 j; // r8
  __int16 v27; // dx
  __int64 v28; // r9
  unsigned int v29; // edi
  unsigned int v30; // esi
  __int64 *v31; // rdx
  __int64 v32; // r8
  __int64 *v33; // rax
  __int64 v34; // rax
  bool v35; // cf
  int v36; // edx
  int v37; // ecx
  unsigned __int64 v38; // [rsp+8h] [rbp-88h]
  int v42; // [rsp+24h] [rbp-6Ch]
  unsigned __int8 v43; // [rsp+2Bh] [rbp-65h]
  int v44; // [rsp+2Ch] [rbp-64h]
  unsigned int v45; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v46; // [rsp+34h] [rbp-5Ch] BYREF
  unsigned __int64 v47; // [rsp+38h] [rbp-58h] BYREF
  _QWORD v48[2]; // [rsp+40h] [rbp-50h] BYREF
  char v49; // [rsp+50h] [rbp-40h]

  v7 = **(_QWORD **)(a1 + 768);
  if ( (unsigned __int8)sub_B2D610(v7, 47) )
    return 0;
  v43 = sub_B2D610(v7, 18);
  if ( v43 )
  {
    return 0;
  }
  else
  {
    v9 = *(_DWORD *)(a3 + 112);
    v10 = v9 & 0x7FFFFFFF;
    if ( *(int *)(*(_QWORD *)(a1 + 920) + 8 * v10) <= 2 )
    {
      v47 = 0;
      v11 = *(_QWORD *)(a1 + 16);
      if ( v9 < 0 )
        v12 = *(_QWORD *)(*(_QWORD *)(v11 + 56) + 16 * v10 + 8);
      else
        v12 = *(_QWORD *)(*(_QWORD *)(v11 + 304) + 8LL * (unsigned int)v9);
      if ( v12 )
      {
        if ( (*(_BYTE *)(v12 + 4) & 8) == 0 )
        {
LABEL_10:
          v13 = *(_QWORD *)(v12 + 16);
          v44 = v9;
          v42 = a2;
          if ( *(_WORD *)(v13 + 68) == 20 )
            goto LABEL_24;
LABEL_11:
          v14 = *(_QWORD *)(a1 + 776);
          v15 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 520LL);
          if ( v15 == sub_2DCA430 )
          {
LABEL_12:
            v16 = v13;
            goto LABEL_14;
          }
          ((void (__fastcall *)(_QWORD *, __int64, unsigned __int64))v15)(v48, v14, v13);
          v19 = (_DWORD *)v48[0];
          v20 = (_DWORD *)v48[1];
          if ( !v49 )
          {
            v13 = *(_QWORD *)(v12 + 16);
            goto LABEL_12;
          }
          while ( 1 )
          {
            if ( (*v19 & 0xFFF00) == 0 && (*v20 & 0xFFF00) == 0 )
            {
              v21 = *(_QWORD *)(v13 + 32);
              v22 = *(_DWORD *)(v21 + 48);
              if ( v44 != v22 )
                goto LABEL_28;
              v22 = *(_DWORD *)(v21 + 8);
              if ( v44 != v22 )
                break;
            }
LABEL_31:
            v16 = *(_QWORD *)(v12 + 16);
            while ( 1 )
            {
LABEL_14:
              v12 = *(_QWORD *)(v12 + 32);
              if ( !v12 )
                goto LABEL_15;
              if ( (*(_BYTE *)(v12 + 4) & 8) == 0 )
              {
                v13 = *(_QWORD *)(v12 + 16);
                if ( v13 != v16 )
                  break;
              }
            }
            if ( *(_WORD *)(v13 + 68) != 20 )
              goto LABEL_11;
LABEL_24:
            v19 = *(_DWORD **)(v13 + 32);
            v20 = v19 + 10;
          }
          v23 = v13;
          v24 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL);
          for ( i = v13; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
            ;
          if ( (*(_DWORD *)(v13 + 44) & 8) != 0 )
          {
            do
              v23 = *(_QWORD *)(v23 + 8);
            while ( (*(_BYTE *)(v23 + 44) & 8) != 0 );
          }
          for ( j = *(_QWORD *)(v23 + 8); j != i; i = *(_QWORD *)(i + 8) )
          {
            v27 = *(_WORD *)(i + 68);
            if ( (unsigned __int16)(v27 - 14) > 4u && v27 != 24 )
              break;
          }
          v28 = *(_QWORD *)(v24 + 128);
          v29 = *(_DWORD *)(v24 + 144);
          if ( v29 )
          {
            v30 = (v29 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
            v31 = (__int64 *)(v28 + 16LL * v30);
            v32 = *v31;
            if ( i == *v31 )
            {
LABEL_45:
              v38 = v31[1] & 0xFFFFFFFFFFFFFFF8LL;
              v33 = (__int64 *)sub_2E09D00((__int64 *)a3, v38 | 4);
              if ( v33 == (__int64 *)(*(_QWORD *)a3 + 24LL * *(unsigned int *)(a3 + 8))
                || (*(_DWORD *)((*v33 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v33 >> 1) & 3) > (*(_DWORD *)(v38 + 24) | 2u) )
              {
LABEL_28:
                if ( (unsigned int)(v22 - 1) > 0x3FFFFFFE )
                  v22 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL) + 4LL * (v22 & 0x7FFFFFFF));
                if ( v42 == v22 )
                {
                  v34 = sub_2E39EA0(*(__int64 **)(a1 + 792), *(_QWORD *)(v13 + 24));
                  v35 = __CFADD__(v47, v34);
                  v47 += v34;
                  if ( v35 )
                    v47 = -1;
                }
                goto LABEL_31;
              }
              goto LABEL_31;
            }
            v36 = 1;
            while ( v32 != -4096 )
            {
              v37 = v36 + 1;
              v30 = (v29 - 1) & (v36 + v30);
              v31 = (__int64 *)(v28 + 16LL * v30);
              v32 = *v31;
              if ( *v31 == i )
                goto LABEL_45;
              v36 = v37;
            }
          }
          v31 = (__int64 *)(v28 + 16LL * v29);
          goto LABEL_45;
        }
        while ( 1 )
        {
          v12 = *(_QWORD *)(v12 + 32);
          if ( !v12 )
            break;
          if ( (*(_BYTE *)(v12 + 4) & 8) == 0 )
            goto LABEL_10;
        }
      }
LABEL_15:
      sub_F02DB0(&v45, qword_5023A68, 0x64u);
      sub_1098CF0(&v47, v45);
      if ( v47 )
      {
        v17 = *(_QWORD *)(a1 + 992);
        v46 = 0;
        LODWORD(v48[0]) = -1;
        sub_2FB1E90(v17, a3);
        sub_2F57410(a1, a2, a5, (__int64)&v47, &v46, (unsigned int *)v48);
        if ( LODWORD(v48[0]) != -1 )
        {
          sub_2F53540(a1, a3, v48[0], 0, a4, v18);
          return 1;
        }
      }
    }
  }
  return v43;
}
