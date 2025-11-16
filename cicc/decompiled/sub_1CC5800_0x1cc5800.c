// Function: sub_1CC5800
// Address: 0x1cc5800
//
__int64 __fastcall sub_1CC5800(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  unsigned __int64 i; // r14
  unsigned __int8 v4; // r12
  __int64 v5; // r11
  __int64 v6; // rax
  _QWORD *v7; // rdx
  unsigned int v8; // r9d
  int v9; // ecx
  unsigned int v10; // eax
  unsigned int v11; // r13d
  __int64 v13; // r10
  unsigned int v14; // eax
  __int64 v15; // r8
  __int64 v16; // rsi
  unsigned __int64 v17; // r11
  unsigned int v18; // eax
  __int64 v19; // rax
  int v20; // ecx
  unsigned int v21; // eax
  __int64 v22; // rax
  unsigned int v23; // esi
  int v24; // eax
  _QWORD *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned int v28; // esi
  __int64 v29; // rax
  unsigned int v30; // [rsp+4h] [rbp-7Ch]
  __int64 v31; // [rsp+8h] [rbp-78h]
  unsigned __int64 v32; // [rsp+10h] [rbp-70h]
  unsigned int v33; // [rsp+10h] [rbp-70h]
  __int64 v34; // [rsp+18h] [rbp-68h]
  __int64 v35; // [rsp+18h] [rbp-68h]
  int v36; // [rsp+18h] [rbp-68h]
  int v37; // [rsp+20h] [rbp-60h]
  __int64 v38; // [rsp+20h] [rbp-60h]
  __int64 v39; // [rsp+20h] [rbp-60h]
  unsigned int v40; // [rsp+28h] [rbp-58h]
  int v41; // [rsp+28h] [rbp-58h]
  unsigned __int64 v42; // [rsp+28h] [rbp-58h]
  unsigned __int64 v43; // [rsp+28h] [rbp-58h]
  __int64 v44; // [rsp+30h] [rbp-50h]
  int v45; // [rsp+30h] [rbp-50h]
  __int64 v46; // [rsp+30h] [rbp-50h]
  __int64 v47; // [rsp+30h] [rbp-50h]
  unsigned int v48; // [rsp+38h] [rbp-48h]
  __int64 v49; // [rsp+38h] [rbp-48h]
  __int64 v50; // [rsp+38h] [rbp-48h]
  __int64 v51; // [rsp+38h] [rbp-48h]
  __int64 v52; // [rsp+38h] [rbp-48h]
  __int64 v53; // [rsp+40h] [rbp-40h]
  int v54; // [rsp+48h] [rbp-38h]
  unsigned __int8 v55; // [rsp+4Fh] [rbp-31h]

  v53 = sub_1632FA0(*(_QWORD *)(a1 + 40));
  v1 = *(_QWORD *)(a1 + 80);
  if ( !v1 )
    BUG();
  v2 = *(_QWORD *)(v1 + 24);
  v55 = 0;
  for ( i = *(_QWORD *)(v1 + 16) & 0xFFFFFFFFFFFFFFF8LL; i != v2; v2 = *(_QWORD *)(v2 + 8) )
  {
    if ( !v2 )
      BUG();
    if ( *(_BYTE *)(v2 - 8) == 53 )
    {
      v4 = sub_15F8F00(v2 - 24);
      if ( v4 )
      {
        v5 = *(_QWORD *)(v2 + 32);
        if ( *(_BYTE *)(v5 + 8) == 14 )
        {
          v6 = *(_QWORD *)(v2 - 48);
          v7 = *(_QWORD **)(v6 + 24);
          v8 = (unsigned int)(1 << *(_WORD *)(v2 - 6)) >> 1;
          if ( *(_DWORD *)(v6 + 32) > 0x40u )
            v7 = (_QWORD *)*v7;
          v9 = (int)v7;
          if ( byte_4FBF0C0 )
          {
            v48 = (unsigned int)(1 << *(_WORD *)(v2 - 6)) >> 1;
            v10 = sub_1CC52F0(v53, v8, (int)v7, *(_QWORD *)(v2 + 32));
            v8 = v48;
            v11 = v10;
          }
          else
          {
            v11 = (unsigned int)(1 << *(_WORD *)(v2 - 6)) >> 1;
            v13 = 1;
            if ( v8 )
            {
LABEL_16:
              v34 = v13;
              v37 = v9;
              v40 = v8;
              v44 = *(_QWORD *)(v5 + 24);
              v49 = *(_QWORD *)(v5 + 32);
              v14 = sub_15A9FE0(v53, v44);
              v8 = v40;
              v9 = v37;
              v15 = 1;
              v16 = v44;
              v13 = v34;
              v17 = v14;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v16 + 8) )
                {
                  case 0:
                  case 8:
                  case 0xA:
                  case 0xC:
                  case 0x10:
                    v29 = *(_QWORD *)(v16 + 32);
                    v16 = *(_QWORD *)(v16 + 24);
                    v15 *= v29;
                    continue;
                  case 1:
                    v26 = 16;
                    goto LABEL_40;
                  case 2:
                    v26 = 32;
                    goto LABEL_40;
                  case 3:
                  case 9:
                    v26 = 64;
                    goto LABEL_40;
                  case 4:
                    v26 = 80;
                    goto LABEL_40;
                  case 5:
                  case 6:
                    v26 = 128;
                    goto LABEL_40;
                  case 7:
                    v33 = v40;
                    v28 = 0;
                    v36 = v37;
                    v39 = v13;
                    v43 = v17;
                    v47 = v15;
                    goto LABEL_47;
                  case 0xB:
                    v26 = *(_DWORD *)(v16 + 8) >> 8;
                    goto LABEL_40;
                  case 0xD:
                    v33 = v40;
                    v36 = v37;
                    v39 = v13;
                    v43 = v17;
                    v47 = v15;
                    v26 = 8LL * *(_QWORD *)sub_15A9930(v53, v16);
                    goto LABEL_44;
                  case 0xE:
                    v30 = v40;
                    v54 = v37;
                    v31 = v34;
                    v32 = v17;
                    v35 = v15;
                    v46 = *(_QWORD *)(v16 + 32);
                    v38 = *(_QWORD *)(v16 + 24);
                    v42 = (unsigned int)sub_15A9FE0(v53, v38);
                    v27 = sub_127FA20(v53, v38);
                    v15 = v35;
                    v17 = v32;
                    v13 = v31;
                    v9 = v54;
                    v8 = v30;
                    v26 = 8 * v42 * v46 * ((v42 + ((unsigned __int64)(v27 + 7) >> 3) - 1) / v42);
                    goto LABEL_40;
                  case 0xF:
                    v33 = v40;
                    v36 = v37;
                    v39 = v13;
                    v28 = *(_DWORD *)(v16 + 8) >> 8;
                    v43 = v17;
                    v47 = v15;
LABEL_47:
                    v26 = 8 * (unsigned int)sub_15A9520(v53, v28);
LABEL_44:
                    v15 = v47;
                    v17 = v43;
                    v13 = v39;
                    v9 = v36;
                    v8 = v33;
LABEL_40:
                    v19 = 8 * v49 * v17 * ((v17 + ((unsigned __int64)(v15 * v26 + 7) >> 3) - 1) / v17);
                    break;
                }
                break;
              }
            }
            else
            {
              v50 = *(_QWORD *)(v2 + 32);
              v41 = (int)v7;
              v18 = sub_15AAE50(v53, v50);
              v5 = v50;
              v8 = 0;
              v9 = v41;
              v13 = 1;
              v11 = v18;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v5 + 8) )
                {
                  case 0:
                  case 8:
                  case 0xA:
                  case 0xC:
                  case 0x10:
                    v22 = *(_QWORD *)(v5 + 32);
                    v5 = *(_QWORD *)(v5 + 24);
                    v13 *= v22;
                    continue;
                  case 1:
                    v19 = 16;
                    goto LABEL_23;
                  case 2:
                    v19 = 32;
                    goto LABEL_23;
                  case 3:
                  case 9:
                    v19 = 64;
                    goto LABEL_23;
                  case 4:
                    v19 = 80;
                    goto LABEL_23;
                  case 5:
                  case 6:
                    v19 = 128;
                    goto LABEL_23;
                  case 7:
                    v23 = 0;
                    v45 = v41;
                    v51 = v13;
                    goto LABEL_35;
                  case 0xB:
                    v19 = *(_DWORD *)(v5 + 8) >> 8;
                    goto LABEL_23;
                  case 0xD:
                    v52 = v13;
                    v25 = (_QWORD *)sub_15A9930(v53, v5);
                    v13 = v52;
                    v9 = v41;
                    v8 = 0;
                    v19 = 8LL * *v25;
                    goto LABEL_23;
                  case 0xE:
                    goto LABEL_16;
                  case 0xF:
                    v45 = v41;
                    v51 = v13;
                    v23 = *(_DWORD *)(v5 + 8) >> 8;
LABEL_35:
                    v24 = sub_15A9520(v53, v23);
                    v13 = v51;
                    v9 = v45;
                    v8 = 0;
                    v19 = (unsigned int)(8 * v24);
                    break;
                }
                break;
              }
            }
LABEL_23:
            v20 = ((unsigned __int64)(v19 * v13 + 7) >> 3) * v9;
            if ( v11 <= 0xF )
            {
              if ( (v20 & 0xF) != 0 )
              {
                v21 = 16;
                while ( 1 )
                {
                  v21 >>= 1;
                  if ( v21 <= v11 )
                    break;
                  if ( (v20 & (v21 - 1)) == 0 )
                  {
                    v11 = v21;
                    break;
                  }
                }
              }
              else
              {
                v11 = 16;
              }
            }
          }
          if ( v8 != v11 )
          {
            sub_15F8A20(v2 - 24, v11);
            v55 = v4;
          }
        }
      }
    }
  }
  return v55;
}
