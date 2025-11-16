// Function: sub_197D610
// Address: 0x197d610
//
bool __fastcall sub_197D610(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // r8
  unsigned __int64 v13; // r12
  __int64 v14; // rax
  unsigned __int64 v15; // r12
  __int64 *v16; // r14
  __int64 *v17; // rax
  __int64 v18; // r13
  unsigned int v19; // ebx
  __int64 v20; // rax
  int v21; // eax
  __int64 v22; // rax
  int v23; // eax
  _QWORD *v24; // rax
  __int64 v25; // rax
  __int64 v26; // r15
  unsigned int v27; // eax
  __int64 v28; // rdi
  __int64 v29; // rdx
  unsigned __int64 v30; // r9
  __int64 v31; // rax
  _QWORD *v32; // rax
  int v33; // eax
  __int64 v34; // rax
  __int64 v35; // [rsp+10h] [rbp-90h]
  __int64 v36; // [rsp+10h] [rbp-90h]
  __int64 v37; // [rsp+18h] [rbp-88h]
  unsigned __int64 v38; // [rsp+18h] [rbp-88h]
  unsigned __int64 v39; // [rsp+18h] [rbp-88h]
  __int64 v40; // [rsp+20h] [rbp-80h]
  __int64 v41; // [rsp+20h] [rbp-80h]
  __int64 v42; // [rsp+20h] [rbp-80h]
  __int64 v44; // [rsp+28h] [rbp-78h]
  __int64 v45; // [rsp+28h] [rbp-78h]
  __int64 v46; // [rsp+28h] [rbp-78h]
  __int64 v47; // [rsp+28h] [rbp-78h]
  __int64 v48; // [rsp+28h] [rbp-78h]
  __int64 v49; // [rsp+30h] [rbp-70h] BYREF
  __int64 v50; // [rsp+38h] [rbp-68h]
  __int64 v51; // [rsp+40h] [rbp-60h]
  int v52; // [rsp+48h] [rbp-58h]
  __int64 v53; // [rsp+50h] [rbp-50h] BYREF
  __int64 v54; // [rsp+58h] [rbp-48h]
  __int64 v55; // [rsp+60h] [rbp-40h]
  int v56; // [rsp+68h] [rbp-38h]

  v5 = *(_QWORD *)(*a1 - 24LL);
  v6 = *(_QWORD *)(a1[1] - 24LL);
  v7 = **(_QWORD **)(*(_QWORD *)v5 + 16LL);
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  if ( sub_385E580(a2, v5, a3, &v49, 0, 1) != 1 )
    goto LABEL_2;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  if ( sub_385E580(a2, v6, a3, &v53, 0, 1) != 1 )
  {
    j___libc_free_0(v54);
LABEL_2:
    j___libc_free_0(v50);
    return 0;
  }
  j___libc_free_0(v54);
  j___libc_free_0(v50);
  v9 = sub_157EB90(*(_QWORD *)(*a1 + 40LL));
  v44 = sub_1632FA0(v9);
  v10 = sub_15A9FE0(v44, v7);
  v11 = v44;
  v12 = 1;
  v13 = v10;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v7 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v22 = *(_QWORD *)(v7 + 32);
        v7 = *(_QWORD *)(v7 + 24);
        v12 *= v22;
        continue;
      case 1:
        v14 = 16;
        break;
      case 2:
        v14 = 32;
        break;
      case 3:
      case 9:
        v14 = 64;
        break;
      case 4:
        v14 = 80;
        break;
      case 5:
      case 6:
        v14 = 128;
        break;
      case 7:
        v45 = v12;
        v21 = sub_15A9520(v11, 0);
        v12 = v45;
        v14 = (unsigned int)(8 * v21);
        break;
      case 0xB:
        v14 = *(_DWORD *)(v7 + 8) >> 8;
        break;
      case 0xD:
        v47 = v12;
        v24 = (_QWORD *)sub_15A9930(v11, v7);
        v12 = v47;
        v14 = 8LL * *v24;
        break;
      case 0xE:
        v25 = *(_QWORD *)(v7 + 32);
        v26 = *(_QWORD *)(v7 + 24);
        v37 = v12;
        v40 = v44;
        v48 = v25;
        v27 = sub_15A9FE0(v11, v26);
        v12 = v37;
        v28 = v40;
        v29 = 1;
        v30 = v27;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v26 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v34 = *(_QWORD *)(v26 + 32);
              v26 = *(_QWORD *)(v26 + 24);
              v29 *= v34;
              continue;
            case 1:
              v31 = 16;
              goto LABEL_26;
            case 2:
              v31 = 32;
              goto LABEL_26;
            case 3:
            case 9:
              v31 = 64;
              goto LABEL_26;
            case 4:
              v31 = 80;
              goto LABEL_26;
            case 5:
            case 6:
              v31 = 128;
              goto LABEL_26;
            case 7:
              JUMPOUT(0x197D9C3);
            case 0xB:
              v31 = *(_DWORD *)(v26 + 8) >> 8;
              goto LABEL_26;
            case 0xD:
              v35 = v29;
              v38 = v30;
              v41 = v12;
              v32 = (_QWORD *)sub_15A9930(v28, v26);
              v12 = v41;
              v30 = v38;
              v29 = v35;
              v31 = 8LL * *v32;
              goto LABEL_26;
            case 0xE:
              sub_12BE0A0(v40, *(_QWORD *)(v26 + 24));
              JUMPOUT(0x197D969);
            case 0xF:
              v36 = v29;
              v39 = v30;
              v42 = v12;
              v33 = sub_15A9520(v28, *(_DWORD *)(v26 + 8) >> 8);
              v12 = v42;
              v30 = v39;
              v29 = v36;
              v31 = (unsigned int)(8 * v33);
LABEL_26:
              v14 = 8 * v48 * v30 * ((v30 + ((unsigned __int64)(v31 * v29 + 7) >> 3) - 1) / v30);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v46 = v12;
        v23 = sub_15A9520(v11, *(_DWORD *)(v7 + 8) >> 8);
        v12 = v46;
        v14 = (unsigned int)(8 * v23);
        break;
    }
    break;
  }
  v15 = (v13 + ((unsigned __int64)(v12 * v14 + 7) >> 3) - 1) / v13 * v13;
  v16 = sub_1494E70(a2, v5, a4, a5);
  v17 = sub_1494E70(a2, v6, a4, a5);
  v18 = *(_QWORD *)(sub_14806B0(*(_QWORD *)(a2 + 112), (__int64)v17, (__int64)v16, 0, 0) + 32);
  v19 = *(_DWORD *)(v18 + 32);
  if ( v19 > 0x40 )
  {
    if ( v19 - (unsigned int)sub_16A57B0(v18 + 24) > 0x40 )
      return 0;
    v20 = **(_QWORD **)(v18 + 24);
  }
  else
  {
    v20 = *(_QWORD *)(v18 + 24);
  }
  return (unsigned int)v15 == v20;
}
