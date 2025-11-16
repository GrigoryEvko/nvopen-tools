// Function: sub_39C9BF0
// Address: 0x39c9bf0
//
unsigned __int64 __fastcall sub_39C9BF0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v4; // rsi
  __int16 v5; // r13
  __int16 v6; // r12
  __int64 v7; // r14
  __int64 v8; // r14
  unsigned __int64 result; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 *v11; // rdx
  __int64 v12; // r14
  __int64 v13; // r14
  __int64 v14; // r14
  __int64 v15; // r14
  __int64 v16; // r14
  __int64 v17; // r14
  __int64 v18; // r14
  __int64 v19; // r14
  __int64 v20; // r14

  v2 = *(_QWORD *)(a1 + 608);
  v4 = (__int64 *)(a1 + 88);
  v5 = *(_WORD *)(v2 + 12);
  v6 = *(_WORD *)(v2 + 14);
  v7 = *(_QWORD *)(v2 + 8);
  switch ( *(_DWORD *)(v2 + 8) )
  {
    case 1:
      v12 = *(_QWORD *)(v2 + 16);
      result = sub_145CDC0(0x18u, v4);
      if ( !result )
        goto LABEL_27;
      *(_DWORD *)(result + 8) = 1;
      v10 = result;
      *(_WORD *)(result + 12) = v5;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v6;
      *(_QWORD *)(result + 16) = v12;
      break;
    case 2:
      v13 = *(_QWORD *)(v2 + 16);
      result = sub_145CDC0(0x18u, v4);
      if ( !result )
        goto LABEL_27;
      *(_DWORD *)(result + 8) = 2;
      v10 = result;
      *(_WORD *)(result + 12) = v5;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v6;
      *(_QWORD *)(result + 16) = v13;
      break;
    case 3:
      v14 = *(_QWORD *)(v2 + 16);
      result = sub_145CDC0(0x18u, v4);
      if ( !result )
        goto LABEL_27;
      *(_DWORD *)(result + 8) = 3;
      v10 = result;
      *(_WORD *)(result + 12) = v5;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v6;
      *(_QWORD *)(result + 16) = v14;
      break;
    case 4:
      v15 = *(_QWORD *)(v2 + 16);
      result = sub_145CDC0(0x18u, v4);
      if ( !result )
        goto LABEL_27;
      *(_DWORD *)(result + 8) = 4;
      v10 = result;
      *(_WORD *)(result + 12) = v5;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v6;
      *(_QWORD *)(result + 16) = v15;
      break;
    case 5:
      v16 = *(_QWORD *)(v2 + 16);
      result = sub_145CDC0(0x18u, v4);
      if ( !result )
        goto LABEL_27;
      *(_DWORD *)(result + 8) = 5;
      v10 = result;
      *(_WORD *)(result + 12) = v5;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v6;
      *(_QWORD *)(result + 16) = v16;
      break;
    case 6:
      v17 = *(_QWORD *)(v2 + 16);
      result = sub_145CDC0(0x18u, v4);
      if ( !result )
        goto LABEL_27;
      *(_DWORD *)(result + 8) = 6;
      v10 = result;
      *(_WORD *)(result + 12) = v5;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v6;
      *(_QWORD *)(result + 16) = v17;
      break;
    case 7:
      v18 = *(_QWORD *)(v2 + 16);
      result = sub_145CDC0(0x18u, v4);
      if ( !result )
        goto LABEL_27;
      *(_DWORD *)(result + 8) = 7;
      v10 = result;
      *(_WORD *)(result + 12) = v5;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v6;
      *(_QWORD *)(result + 16) = v18;
      break;
    case 8:
      v19 = *(_QWORD *)(v2 + 16);
      result = sub_145CDC0(0x18u, v4);
      if ( !result )
        goto LABEL_27;
      *(_DWORD *)(result + 8) = 8;
      v10 = result;
      *(_WORD *)(result + 12) = v5;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v6;
      *(_QWORD *)(result + 16) = v19;
      break;
    case 9:
      v20 = *(_QWORD *)(v2 + 16);
      result = sub_145CDC0(0x18u, v4);
      if ( !result )
        goto LABEL_27;
      *(_DWORD *)(result + 8) = 9;
      v10 = result;
      *(_WORD *)(result + 12) = v5;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v6;
      *(_QWORD *)(result + 16) = v20;
      break;
    case 0xA:
      v8 = *(_QWORD *)(v2 + 16);
      result = sub_145CDC0(0x18u, v4);
      if ( !result )
        goto LABEL_27;
      *(_DWORD *)(result + 8) = 10;
      v10 = result;
      *(_WORD *)(result + 12) = v5;
      *(_QWORD *)result = result | 4;
      *(_WORD *)(result + 14) = v6;
      *(_QWORD *)(result + 16) = v8;
      break;
    default:
      result = sub_145CDC0(0x18u, v4);
      if ( result )
      {
        *(_QWORD *)(result + 8) = v7;
        v10 = result;
        *(_QWORD *)result = result | 4;
      }
      else
      {
LABEL_27:
        v10 = 0;
      }
      break;
  }
  v11 = *(unsigned __int64 **)(a2 + 8);
  if ( v11 )
  {
    *(_QWORD *)result = *v11;
    result = v10 & 0xFFFFFFFFFFFFFFFBLL;
    *v11 = v10 & 0xFFFFFFFFFFFFFFFBLL;
  }
  *(_QWORD *)(a2 + 8) = v10;
  return result;
}
