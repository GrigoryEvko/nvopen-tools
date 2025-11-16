// Function: sub_140BFD0
// Address: 0x140bfd0
//
__int64 __fastcall sub_140BFD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v7; // r15
  unsigned int v8; // eax
  __int64 v9; // rdi
  __int64 v10; // rcx
  unsigned __int64 v11; // r14
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned int v14; // ecx
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // r13
  unsigned int v19; // eax
  unsigned int v20; // eax
  int v21; // eax
  __int64 v22; // rax
  int v23; // eax
  unsigned int v24; // eax
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rdi
  unsigned __int64 v28; // r15
  _QWORD *v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rax
  _QWORD *v32; // rax
  int v33; // eax
  __int64 v34; // rax
  __int64 v35; // [rsp+0h] [rbp-80h]
  __int64 v36; // [rsp+0h] [rbp-80h]
  __int64 v37; // [rsp+8h] [rbp-78h]
  __int64 v38; // [rsp+8h] [rbp-78h]
  __int64 v39; // [rsp+8h] [rbp-78h]
  __int64 v40; // [rsp+8h] [rbp-78h]
  __int64 v41; // [rsp+10h] [rbp-70h]
  __int64 v42; // [rsp+10h] [rbp-70h]
  __int64 v43; // [rsp+18h] [rbp-68h]
  __int64 v44; // [rsp+18h] [rbp-68h]
  __int64 v45; // [rsp+18h] [rbp-68h]
  __int64 v46; // [rsp+18h] [rbp-68h]
  __int64 v47; // [rsp+18h] [rbp-68h]
  unsigned __int64 v48; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v49; // [rsp+28h] [rbp-58h]
  unsigned __int64 v50; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v51; // [rsp+38h] [rbp-48h]
  __int64 v52; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v53; // [rsp+48h] [rbp-38h]

  if ( (unsigned __int8)sub_15E0300(a3) )
  {
    v7 = *(_QWORD *)(*(_QWORD *)a3 + 24LL);
    v43 = *(_QWORD *)a2;
    v8 = sub_15A9FE0(*(_QWORD *)a2, v7);
    v9 = v43;
    v10 = 1;
    v11 = v8;
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
          v10 *= v22;
          continue;
        case 1:
          v12 = 16;
          break;
        case 2:
          v12 = 32;
          break;
        case 3:
        case 9:
          v12 = 64;
          break;
        case 4:
          v12 = 80;
          break;
        case 5:
        case 6:
          v12 = 128;
          break;
        case 7:
          v44 = v10;
          v21 = sub_15A9520(v9, 0);
          v10 = v44;
          v12 = (unsigned int)(8 * v21);
          break;
        case 0xB:
          v12 = *(_DWORD *)(v7 + 8) >> 8;
          break;
        case 0xD:
          v47 = v10;
          v29 = (_QWORD *)sub_15A9930(v9, v7);
          v10 = v47;
          v12 = 8LL * *v29;
          break;
        case 0xE:
          v35 = v10;
          v41 = v43;
          v37 = *(_QWORD *)(v7 + 24);
          v46 = *(_QWORD *)(v7 + 32);
          v24 = sub_15A9FE0(v9, v37);
          v10 = v35;
          v25 = v37;
          v26 = 1;
          v27 = v41;
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
                v34 = *(_QWORD *)(v25 + 32);
                v25 = *(_QWORD *)(v25 + 24);
                v26 *= v34;
                continue;
              case 1:
                v30 = 16;
                goto LABEL_36;
              case 2:
                v30 = 32;
                goto LABEL_36;
              case 3:
              case 9:
                v30 = 64;
                goto LABEL_36;
              case 4:
                v30 = 80;
                goto LABEL_36;
              case 5:
              case 6:
                v30 = 128;
                goto LABEL_36;
              case 7:
                JUMPOUT(0x140C357);
              case 0xB:
                v30 = *(_DWORD *)(v25 + 8) >> 8;
                goto LABEL_36;
              case 0xD:
                v39 = v26;
                v32 = (_QWORD *)sub_15A9930(v41, v25);
                v10 = v35;
                v26 = v39;
                v30 = 8LL * *v32;
                goto LABEL_36;
              case 0xE:
                v36 = v26;
                v38 = v10;
                v42 = *(_QWORD *)(v25 + 32);
                v31 = sub_12BE0A0(v27, *(_QWORD *)(v25 + 24));
                v10 = v38;
                v26 = v36;
                v30 = 8 * v42 * v31;
                goto LABEL_36;
              case 0xF:
                v40 = v26;
                v33 = sub_15A9520(v41, *(_DWORD *)(v25 + 8) >> 8);
                v10 = v35;
                v26 = v40;
                v30 = (unsigned int)(8 * v33);
LABEL_36:
                v12 = 8 * v28 * v46 * ((v28 + ((unsigned __int64)(v30 * v26 + 7) >> 3) - 1) / v28);
                break;
            }
            break;
          }
          break;
        case 0xF:
          v45 = v10;
          v23 = sub_15A9520(v9, *(_DWORD *)(v7 + 8) >> 8);
          v10 = v45;
          v12 = (unsigned int)(8 * v23);
          break;
      }
      break;
    }
    v13 = v11 + ((unsigned __int64)(v12 * v10 + 7) >> 3) - 1;
    v14 = *(_DWORD *)(a2 + 20);
    v15 = v13 % v11;
    v49 = v14;
    v16 = v11 * (v13 / v11);
    v17 = v16;
    if ( v14 > 0x40 )
      sub_16A4EF0(&v48, v16, 0);
    else
      v48 = v16 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v14);
    v18 = (unsigned int)sub_15E0370(a3, v17, v15);
    v51 = v49;
    if ( v49 > 0x40 )
      sub_16A4FD0(&v50, &v48);
    else
      v50 = v48;
    sub_140B7A0((__int64)&v52, a2, (__int64)&v50, v18);
    v19 = v53;
    v53 = 0;
    *(_DWORD *)(a1 + 8) = v19;
    *(_QWORD *)a1 = v52;
    v20 = *(_DWORD *)(a2 + 32);
    *(_DWORD *)(a1 + 24) = v20;
    if ( v20 > 0x40 )
    {
      sub_16A4FD0(a1 + 16, a2 + 24);
      if ( v53 > 0x40 && v52 )
        j_j___libc_free_0_0(v52);
    }
    else
    {
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 24);
    }
    if ( v51 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
    if ( v49 > 0x40 && v48 )
      j_j___libc_free_0_0(v48);
  }
  else
  {
    *(_DWORD *)(a1 + 8) = 1;
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = 1;
    *(_QWORD *)(a1 + 16) = 0;
  }
  return a1;
}
