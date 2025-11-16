// Function: sub_1C7E130
// Address: 0x1c7e130
//
unsigned __int64 __fastcall sub_1C7E130(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6)
{
  unsigned __int64 result; // rax
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // r8
  unsigned __int64 v13; // r9
  __int64 v14; // rbx
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rax
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // r8
  unsigned int v20; // edx
  unsigned __int64 *v21; // rax
  __int64 v22; // rax
  unsigned int v23; // esi
  unsigned int v24; // eax
  __int64 v25; // rsi
  __int64 v26; // r11
  __int64 v27; // rcx
  unsigned __int64 v28; // r10
  __int64 v29; // rax
  unsigned int v30; // esi
  __int64 v31; // rax
  unsigned __int64 v32; // [rsp+38h] [rbp-78h]
  unsigned __int64 v33; // [rsp+40h] [rbp-70h]
  __int64 v34; // [rsp+40h] [rbp-70h]
  unsigned __int64 v35; // [rsp+48h] [rbp-68h]
  unsigned __int64 v36; // [rsp+50h] [rbp-60h]
  unsigned __int64 v37; // [rsp+50h] [rbp-60h]
  __int64 v38; // [rsp+50h] [rbp-60h]
  __int64 v39; // [rsp+58h] [rbp-58h]
  __int64 v40; // [rsp+58h] [rbp-58h]
  unsigned __int64 v41; // [rsp+58h] [rbp-58h]
  unsigned __int64 v43; // [rsp+60h] [rbp-50h]
  __int64 v44; // [rsp+60h] [rbp-50h]
  unsigned __int64 v45; // [rsp+60h] [rbp-50h]
  __int64 v47; // [rsp+70h] [rbp-40h]

  v47 = sub_15A9930(a2, a1);
  result = *(unsigned int *)(a1 + 12);
  if ( (_DWORD)result )
  {
    v9 = a6;
    v10 = 0;
    v11 = a5;
    while ( 1 )
    {
      v13 = v9;
      v14 = 1;
      v9 = *(_QWORD *)(v47 + 8LL * (unsigned int)v10 + 16) + a4;
      v15 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v10);
      v16 = v15;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v16 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v22 = *(_QWORD *)(v16 + 32);
            v16 = *(_QWORD *)(v16 + 24);
            v14 *= v22;
            continue;
          case 1:
            v17 = 16;
            goto LABEL_6;
          case 2:
            v17 = 32;
            goto LABEL_6;
          case 3:
          case 9:
            v17 = 64;
            goto LABEL_6;
          case 4:
            v17 = 80;
            goto LABEL_6;
          case 5:
          case 6:
            v17 = 128;
            goto LABEL_6;
          case 7:
            v36 = v13;
            v23 = 0;
            v39 = 8 * v10;
            v43 = v11;
            goto LABEL_22;
          case 0xB:
            v17 = *(_DWORD *)(v16 + 8) >> 8;
            goto LABEL_6;
          case 0xD:
            v36 = v13;
            v39 = 8 * v10;
            v43 = v11;
            v17 = 8LL * *(_QWORD *)sub_15A9930(a2, v16);
            goto LABEL_23;
          case 0xE:
            v33 = v13;
            v37 = v11;
            v40 = *(_QWORD *)(v16 + 24);
            v44 = *(_QWORD *)(v16 + 32);
            v24 = sub_15A9FE0(a2, v40);
            v25 = v40;
            v13 = v33;
            v26 = 1;
            v27 = 8 * v10;
            v11 = v37;
            v28 = v24;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v25 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v31 = *(_QWORD *)(v25 + 32);
                  v25 = *(_QWORD *)(v25 + 24);
                  v26 *= v31;
                  continue;
                case 1:
                  v29 = 16;
                  goto LABEL_31;
                case 2:
                  v29 = 32;
                  goto LABEL_31;
                case 3:
                case 9:
                  v29 = 64;
                  goto LABEL_31;
                case 4:
                  v29 = 80;
                  goto LABEL_31;
                case 5:
                case 6:
                  v29 = 128;
                  goto LABEL_31;
                case 7:
                  v32 = v28;
                  v30 = 0;
                  v34 = v26;
                  v35 = v13;
                  v38 = 8 * v10;
                  v41 = v11;
                  goto LABEL_34;
                case 0xB:
                  v29 = *(_DWORD *)(v25 + 8) >> 8;
                  goto LABEL_31;
                case 0xD:
                  v32 = v28;
                  v34 = v26;
                  v35 = v13;
                  v38 = 8 * v10;
                  v41 = v11;
                  v29 = 8LL * *(_QWORD *)sub_15A9930(a2, v25);
                  goto LABEL_35;
                case 0xE:
                  JUMPOUT(0x1C7E4BB);
                case 0xF:
                  v32 = v28;
                  v34 = v26;
                  v35 = v13;
                  v30 = *(_DWORD *)(v25 + 8) >> 8;
                  v38 = 8 * v10;
                  v41 = v11;
LABEL_34:
                  v29 = 8 * (unsigned int)sub_15A9520(a2, v30);
LABEL_35:
                  v11 = v41;
                  v27 = v38;
                  v13 = v35;
                  v26 = v34;
                  v28 = v32;
LABEL_31:
                  v15 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + v27);
                  v17 = 8 * v28 * v44 * ((v28 + ((unsigned __int64)(v26 * v29 + 7) >> 3) - 1) / v28);
                  break;
              }
              goto LABEL_6;
            }
          case 0xF:
            v36 = v13;
            v39 = 8 * v10;
            v43 = v11;
            v23 = *(_DWORD *)(v16 + 8) >> 8;
LABEL_22:
            v17 = 8 * (unsigned int)sub_15A9520(a2, v23);
LABEL_23:
            v11 = v43;
            v13 = v36;
            v15 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + v39);
LABEL_6:
            v18 = (unsigned __int64)(v17 * v14 + 7) >> 3;
            if ( *(_BYTE *)(v15 + 8) == 13 )
            {
              sub_1C7E130(v15, a2, a3, v9, v11, v13);
LABEL_8:
              result = *(unsigned int *)(a1 + 12);
              if ( result <= ++v10 )
                return result;
              goto LABEL_9;
            }
            v19 = v13 + v11;
            if ( v19 >= v9 )
              goto LABEL_8;
            v20 = *(_DWORD *)(a3 + 8);
            if ( v20 >= *(_DWORD *)(a3 + 12) )
            {
              v45 = v19;
              sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v19, v13);
              v19 = v45;
              v20 = *(_DWORD *)(a3 + 8);
            }
            v21 = (unsigned __int64 *)(*(_QWORD *)a3 + 16LL * v20);
            if ( v21 )
            {
              *v21 = v19;
              v21[1] = v9 - v19;
              v20 = *(_DWORD *)(a3 + 8);
            }
            ++v10;
            *(_DWORD *)(a3 + 8) = v20 + 1;
            result = *(unsigned int *)(a1 + 12);
            if ( result <= v10 )
              return result;
LABEL_9:
            v11 = v18;
            break;
        }
        break;
      }
    }
  }
  return result;
}
