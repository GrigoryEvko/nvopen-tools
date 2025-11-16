// Function: sub_3248F80
// Address: 0x3248f80
//
__int64 __fastcall sub_3248F80(unsigned __int64 **a1, __int64 *a2, __int64 *a3)
{
  __int16 v3; // r13
  __int16 v4; // r12
  __int64 v5; // r14
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v10; // r14
  __int64 v11; // r14
  __int64 v12; // r14
  __int64 v13; // r14
  __int64 v14; // r14
  __int64 v15; // r14
  __int64 v16; // r14
  __int64 v17; // r14
  __int64 v18; // r14
  __int64 v19; // r14
  __int64 v20; // r14

  v3 = *((_WORD *)a3 + 2);
  v4 = *((_WORD *)a3 + 3);
  v5 = *a3;
  switch ( *(_DWORD *)a3 )
  {
    case 1:
      v10 = a3[1];
      v7 = sub_A777F0(0x18u, a2);
      if ( !v7 )
        goto LABEL_31;
      *(_DWORD *)(v7 + 8) = 1;
      v8 = v7;
      *(_WORD *)(v7 + 12) = v3;
      *(_QWORD *)v7 = v7 | 4;
      *(_WORD *)(v7 + 14) = v4;
      *(_QWORD *)(v7 + 16) = v10;
      break;
    case 2:
      v11 = a3[1];
      v7 = sub_A777F0(0x18u, a2);
      if ( !v7 )
        goto LABEL_31;
      *(_DWORD *)(v7 + 8) = 2;
      v8 = v7;
      *(_WORD *)(v7 + 12) = v3;
      *(_QWORD *)v7 = v7 | 4;
      *(_WORD *)(v7 + 14) = v4;
      *(_QWORD *)(v7 + 16) = v11;
      break;
    case 3:
      v12 = a3[1];
      v7 = sub_A777F0(0x18u, a2);
      if ( !v7 )
        goto LABEL_31;
      *(_DWORD *)(v7 + 8) = 3;
      v8 = v7;
      *(_WORD *)(v7 + 12) = v3;
      *(_QWORD *)v7 = v7 | 4;
      *(_WORD *)(v7 + 14) = v4;
      *(_QWORD *)(v7 + 16) = v12;
      break;
    case 4:
      v13 = a3[1];
      v7 = sub_A777F0(0x18u, a2);
      if ( !v7 )
        goto LABEL_31;
      *(_DWORD *)(v7 + 8) = 4;
      v8 = v7;
      *(_WORD *)(v7 + 12) = v3;
      *(_QWORD *)v7 = v7 | 4;
      *(_WORD *)(v7 + 14) = v4;
      *(_QWORD *)(v7 + 16) = v13;
      break;
    case 5:
      v14 = a3[1];
      v7 = sub_A777F0(0x18u, a2);
      if ( !v7 )
        goto LABEL_31;
      *(_DWORD *)(v7 + 8) = 5;
      v8 = v7;
      *(_WORD *)(v7 + 12) = v3;
      *(_QWORD *)v7 = v7 | 4;
      *(_WORD *)(v7 + 14) = v4;
      *(_QWORD *)(v7 + 16) = v14;
      break;
    case 6:
      v15 = a3[1];
      v7 = sub_A777F0(0x18u, a2);
      if ( !v7 )
        goto LABEL_31;
      *(_DWORD *)(v7 + 8) = 6;
      v8 = v7;
      *(_WORD *)(v7 + 12) = v3;
      *(_QWORD *)v7 = v7 | 4;
      *(_WORD *)(v7 + 14) = v4;
      *(_QWORD *)(v7 + 16) = v15;
      break;
    case 7:
      v16 = a3[1];
      v7 = sub_A777F0(0x18u, a2);
      if ( !v7 )
        goto LABEL_31;
      *(_DWORD *)(v7 + 8) = 7;
      v8 = v7;
      *(_WORD *)(v7 + 12) = v3;
      *(_QWORD *)v7 = v7 | 4;
      *(_WORD *)(v7 + 14) = v4;
      *(_QWORD *)(v7 + 16) = v16;
      break;
    case 8:
      v17 = a3[1];
      v7 = sub_A777F0(0x18u, a2);
      if ( !v7 )
        goto LABEL_31;
      *(_DWORD *)(v7 + 8) = 8;
      v8 = v7;
      *(_WORD *)(v7 + 12) = v3;
      *(_QWORD *)v7 = v7 | 4;
      *(_WORD *)(v7 + 14) = v4;
      *(_QWORD *)(v7 + 16) = v17;
      break;
    case 9:
      v18 = a3[1];
      v7 = sub_A777F0(0x18u, a2);
      if ( !v7 )
        goto LABEL_31;
      *(_DWORD *)(v7 + 8) = 9;
      v8 = v7;
      *(_WORD *)(v7 + 12) = v3;
      *(_QWORD *)v7 = v7 | 4;
      *(_WORD *)(v7 + 14) = v4;
      *(_QWORD *)(v7 + 16) = v18;
      break;
    case 0xA:
      v19 = a3[1];
      v7 = sub_A777F0(0x18u, a2);
      if ( !v7 )
        goto LABEL_31;
      *(_DWORD *)(v7 + 8) = 10;
      v8 = v7;
      *(_WORD *)(v7 + 12) = v3;
      *(_QWORD *)v7 = v7 | 4;
      *(_WORD *)(v7 + 14) = v4;
      *(_QWORD *)(v7 + 16) = v19;
      break;
    case 0xB:
      v20 = a3[1];
      v7 = sub_A777F0(0x18u, a2);
      if ( !v7 )
        goto LABEL_31;
      *(_DWORD *)(v7 + 8) = 11;
      v8 = v7;
      *(_WORD *)(v7 + 12) = v3;
      *(_QWORD *)v7 = v7 | 4;
      *(_WORD *)(v7 + 14) = v4;
      *(_QWORD *)(v7 + 16) = v20;
      break;
    case 0xC:
      v6 = a3[1];
      v7 = sub_A777F0(0x18u, a2);
      if ( !v7 )
        goto LABEL_31;
      *(_DWORD *)(v7 + 8) = 12;
      v8 = v7;
      *(_WORD *)(v7 + 12) = v3;
      *(_QWORD *)v7 = v7 | 4;
      *(_WORD *)(v7 + 14) = v4;
      *(_QWORD *)(v7 + 16) = v6;
      break;
    default:
      v7 = sub_A777F0(0x18u, a2);
      if ( v7 )
      {
        *(_QWORD *)(v7 + 8) = v5;
        v8 = v7;
        *(_QWORD *)v7 = v7 | 4;
      }
      else
      {
LABEL_31:
        v8 = 0;
      }
      break;
  }
  if ( *a1 )
  {
    *(_QWORD *)v7 = **a1;
    **a1 = v8 & 0xFFFFFFFFFFFFFFFBLL;
  }
  *a1 = (unsigned __int64 *)v8;
  return v8;
}
