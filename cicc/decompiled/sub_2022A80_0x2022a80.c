// Function: sub_2022A80
// Address: 0x2022a80
//
__int64 __fastcall sub_2022A80(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  unsigned int v9; // eax
  __int16 v10; // cx
  char v11; // al
  unsigned int v12; // ebx
  __int64 v13; // rcx
  __int64 v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // r8
  char v18; // cl
  __int64 v19; // rax
  unsigned __int8 *v20; // rax
  __int64 v21; // r8
  unsigned int v22; // ecx
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rcx
  char *v26; // rdx
  char v27; // al
  __int64 v28; // rdx
  char v29; // al
  __int64 v30; // rdx
  __int64 v31; // r8
  const __m128i *v32; // r9
  __int64 v33; // r14
  __int64 v35; // [rsp+8h] [rbp-B8h]
  __int64 v36; // [rsp+8h] [rbp-B8h]
  __int128 v37; // [rsp+10h] [rbp-B0h]
  unsigned int v38; // [rsp+24h] [rbp-9Ch]
  _QWORD *v39; // [rsp+28h] [rbp-98h]
  __int64 v40; // [rsp+30h] [rbp-90h]
  __int64 v41; // [rsp+38h] [rbp-88h]
  char v42[8]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v43; // [rsp+48h] [rbp-78h]
  char v44[8]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v45; // [rsp+58h] [rbp-68h]
  __int64 v46; // [rsp+60h] [rbp-60h] BYREF
  int v47; // [rsp+68h] [rbp-58h]
  __m128i v48; // [rsp+70h] [rbp-50h] BYREF
  __int64 v49; // [rsp+80h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 104);
  v39 = *(_QWORD **)(a1 + 8);
  v49 = *(_QWORD *)(v8 + 56);
  v9 = *(unsigned __int16 *)(v8 + 32);
  v10 = *(_WORD *)(v8 + 34);
  v48 = _mm_loadu_si128((const __m128i *)(v8 + 40));
  v38 = v9;
  v11 = *(_BYTE *)(a2 + 88);
  v12 = 1 << v10;
  v13 = *(_QWORD *)(a2 + 96);
  v44[0] = v11;
  v45 = v13;
  if ( v11 )
  {
    switch ( v11 )
    {
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
        v18 = 2;
        break;
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        v18 = 3;
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        v18 = 4;
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v18 = 5;
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        v18 = 6;
        break;
      case 55:
        v18 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v18 = 8;
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        v18 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v18 = 10;
        break;
    }
    v15 = v39;
    v19 = 0;
  }
  else
  {
    LOBYTE(v14) = sub_1F596B0((__int64)v44);
    v15 = *(_QWORD **)(a1 + 8);
    v8 = *(_QWORD *)(a2 + 104);
    v17 = v16;
    v18 = v14;
    a3 = v14;
    v19 = v17;
  }
  v41 = v19;
  LOBYTE(a3) = v18;
  v40 = a3;
  v20 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL));
  v21 = *((_QWORD *)v20 + 1);
  v22 = *v20;
  v46 = 0;
  v47 = 0;
  *(_QWORD *)&v37 = sub_1D2B300(v15, 0x30u, (__int64)&v46, v22, v21, a6);
  *((_QWORD *)&v37 + 1) = v23;
  if ( v46 )
    sub_161E7C0((__int64)&v46, v46);
  v24 = *(_QWORD *)(a2 + 72);
  v25 = *(_QWORD *)(a2 + 32);
  v46 = v24;
  if ( v24 )
  {
    v35 = v25;
    sub_1623A60((__int64)&v46, v24, 2);
    v25 = v35;
  }
  v26 = *(char **)(a2 + 40);
  v47 = *(_DWORD *)(a2 + 64);
  v27 = *v26;
  v28 = *((_QWORD *)v26 + 1);
  v42[0] = v27;
  v43 = v28;
  if ( v27 )
  {
    switch ( v27 )
    {
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
        v29 = 2;
        break;
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        v29 = 3;
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        v29 = 4;
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v29 = 5;
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        v29 = 6;
        break;
      case 55:
        v29 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v29 = 8;
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        v29 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v29 = 10;
        break;
      default:
        BUG();
    }
    v31 = 0;
  }
  else
  {
    v36 = v25;
    v29 = sub_1F596B0((__int64)v42);
    v25 = v36;
    v31 = v30;
  }
  v33 = sub_1D264C0(
          v39,
          0,
          (*(_BYTE *)(a2 + 27) >> 2) & 3,
          v29,
          v31,
          (__int64)&v46,
          *(_OWORD *)v25,
          *(_QWORD *)(v25 + 40),
          *(_QWORD *)(v25 + 48),
          v37,
          *(_OWORD *)v8,
          *(_QWORD *)(v8 + 16),
          v40,
          v41,
          v12 >> 1,
          v38,
          (__int64)&v48,
          0);
  if ( v46 )
    sub_161E7C0((__int64)&v46, v46);
  sub_2013400(a1, a2, 1, v33, (__m128i *)1, v32);
  return v33;
}
