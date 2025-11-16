// Function: sub_213C0E0
// Address: 0x213c0e0
//
__int64 __fastcall sub_213C0E0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __m128i a5, double a6, __m128i a7)
{
  int v9; // edx
  unsigned int v11; // edx
  int v12; // edx
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // r14
  __int64 v18; // rdx
  __int64 v19; // rax
  char v20; // di
  __int64 v21; // rdx
  __int64 v22; // rax
  char v23; // di
  __int64 v24; // rdx
  __int64 v25; // rax
  char v26; // di
  __int64 v27; // rdx
  unsigned int v28; // eax
  __int64 v29; // rax
  char v30; // di
  __int64 v31; // rdx
  unsigned int v32; // eax
  int v33; // edx
  unsigned int v34; // edx
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // [rsp+8h] [rbp-108h]
  __int64 v40; // [rsp+10h] [rbp-100h]
  int v41; // [rsp+18h] [rbp-F8h]
  __int64 v42; // [rsp+18h] [rbp-F8h]
  __int64 v43; // [rsp+20h] [rbp-F0h]
  int v44; // [rsp+20h] [rbp-F0h]
  int v45; // [rsp+28h] [rbp-E8h]
  unsigned int v46; // [rsp+28h] [rbp-E8h]
  __int64 v47; // [rsp+30h] [rbp-E0h]
  __int64 v48; // [rsp+38h] [rbp-D8h]
  char v49[8]; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v50; // [rsp+C8h] [rbp-48h]
  char v51[8]; // [rsp+D0h] [rbp-40h] BYREF
  __int64 v52; // [rsp+D8h] [rbp-38h]

  if ( a4 != 17 )
  {
    if ( a4 <= 0x11 )
    {
      *(_QWORD *)a2 = sub_2139210(a1, *(_QWORD *)a2, *(_QWORD *)(a2 + 8), a5, a6, a7);
      *(_DWORD *)(a2 + 8) = v12;
      *(_QWORD *)a3 = sub_2139210(a1, *(_QWORD *)a3, *(_QWORD *)(a3 + 8), a5, a6, a7);
      *(_DWORD *)(a3 + 8) = v13;
      return v13;
    }
    if ( a4 <= 0x15 )
    {
      *(_QWORD *)a2 = sub_2139100(a1, *(_QWORD *)a2, *(_QWORD *)(a2 + 8), *(double *)a5.m128i_i64, a6, a7);
      *(_DWORD *)(a2 + 8) = v9;
      *(_QWORD *)a3 = sub_2139100(a1, *(_QWORD *)a3, *(_QWORD *)(a3 + 8), *(double *)a5.m128i_i64, a6, a7);
      *(_DWORD *)(a3 + 8) = v11;
      return v11;
    }
  }
  v14 = sub_2138AD0(a1, *(_QWORD *)a2, *(_QWORD *)(a2 + 8));
  v48 = v15;
  v16 = v14;
  v17 = sub_2138AD0(a1, *(_QWORD *)a3, *(_QWORD *)(a3 + 8));
  v47 = v18;
  v19 = *(_QWORD *)(v16 + 40) + 16LL * (unsigned int)v48;
  v20 = *(_BYTE *)v19;
  v21 = *(_QWORD *)(v19 + 8);
  v51[0] = v20;
  v52 = v21;
  if ( v20 )
  {
    if ( (unsigned __int8)(v20 - 14) <= 0x5Fu )
    {
      switch ( v20 )
      {
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
          v20 = 3;
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
          v20 = 4;
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
          v20 = 5;
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
          v20 = 6;
          break;
        case 55:
          v20 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v20 = 8;
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
          v20 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v20 = 10;
          break;
        default:
          v20 = 2;
          break;
      }
    }
  }
  else
  {
    v43 = v21;
    if ( !sub_1F58D20((__int64)v51) )
    {
      v49[0] = 0;
      v50 = v43;
LABEL_9:
      v45 = sub_1F58D40((__int64)v49);
      goto LABEL_10;
    }
    v49[0] = sub_1F596B0((__int64)v51);
    v20 = v49[0];
    v50 = v35;
    if ( !v49[0] )
      goto LABEL_9;
  }
  v45 = sub_2127930(v20);
LABEL_10:
  v46 = v45 + 1 - sub_1D23330(*(_QWORD *)(a1 + 8), v16, v48, 0);
  v22 = *(_QWORD *)(v17 + 40) + 16LL * (unsigned int)v47;
  v23 = *(_BYTE *)v22;
  v24 = *(_QWORD *)(v22 + 8);
  v51[0] = v23;
  v52 = v24;
  if ( v23 )
  {
    if ( (unsigned __int8)(v23 - 14) <= 0x5Fu )
    {
      switch ( v23 )
      {
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
          v23 = 3;
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
          v23 = 4;
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
          v23 = 5;
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
          v23 = 6;
          break;
        case 55:
          v23 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v23 = 8;
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
          v23 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v23 = 10;
          break;
        default:
          v23 = 2;
          break;
      }
    }
    goto LABEL_12;
  }
  v42 = v24;
  if ( sub_1F58D20((__int64)v51) )
  {
    v49[0] = sub_1F596B0((__int64)v51);
    v23 = v49[0];
    v50 = v37;
    if ( v49[0] )
    {
LABEL_12:
      v41 = sub_2127930(v23);
      goto LABEL_13;
    }
  }
  else
  {
    v49[0] = 0;
    v50 = v42;
  }
  v41 = sub_1F58D40((__int64)v49);
LABEL_13:
  v44 = sub_1D23330(*(_QWORD *)(a1 + 8), v17, v47, 0);
  v25 = *(_QWORD *)(*(_QWORD *)a2 + 40LL) + 16LL * *(unsigned int *)(a2 + 8);
  v26 = *(_BYTE *)v25;
  v27 = *(_QWORD *)(v25 + 8);
  v51[0] = v26;
  v52 = v27;
  if ( v26 )
  {
    if ( (unsigned __int8)(v26 - 14) <= 0x5Fu )
    {
      switch ( v26 )
      {
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
          v26 = 3;
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
          v26 = 4;
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
          v26 = 5;
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
          v26 = 6;
          break;
        case 55:
          v26 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v26 = 8;
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
          v26 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v26 = 10;
          break;
        default:
          v26 = 2;
          break;
      }
    }
    goto LABEL_15;
  }
  v39 = v27;
  if ( sub_1F58D20((__int64)v51) )
  {
    v49[0] = sub_1F596B0((__int64)v51);
    v26 = v49[0];
    v50 = v36;
    if ( v49[0] )
    {
LABEL_15:
      v28 = sub_2127930(v26);
      goto LABEL_16;
    }
  }
  else
  {
    v49[0] = 0;
    v50 = v39;
  }
  v28 = sub_1F58D40((__int64)v49);
LABEL_16:
  if ( v46 > v28 )
    goto LABEL_24;
  v29 = *(_QWORD *)(*(_QWORD *)a3 + 40LL) + 16LL * *(unsigned int *)(a3 + 8);
  v30 = *(_BYTE *)v29;
  v31 = *(_QWORD *)(v29 + 8);
  v51[0] = v30;
  v52 = v31;
  if ( v30 )
  {
    if ( (unsigned __int8)(v30 - 14) <= 0x5Fu )
    {
      switch ( v30 )
      {
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
          v30 = 3;
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
          v30 = 4;
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
          v30 = 5;
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
          v30 = 6;
          break;
        case 55:
          v30 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v30 = 8;
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
          v30 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v30 = 10;
          break;
        default:
          v30 = 2;
          break;
      }
    }
    goto LABEL_19;
  }
  v40 = v31;
  if ( sub_1F58D20((__int64)v51) )
  {
    v49[0] = sub_1F596B0((__int64)v51);
    v30 = v49[0];
    v50 = v38;
    if ( v49[0] )
    {
LABEL_19:
      v32 = sub_2127930(v30);
      goto LABEL_20;
    }
  }
  else
  {
    v49[0] = 0;
    v50 = v40;
  }
  v32 = sub_1F58D40((__int64)v49);
LABEL_20:
  if ( v41 + 1 - v44 <= v32 )
  {
    *(_QWORD *)a2 = v16;
    *(_DWORD *)(a2 + 8) = v48;
    *(_QWORD *)a3 = v17;
    *(_DWORD *)(a3 + 8) = v47;
    return (unsigned int)v47;
  }
LABEL_24:
  *(_QWORD *)a2 = sub_2139210(a1, *(_QWORD *)a2, *(_QWORD *)(a2 + 8), a5, a6, a7);
  *(_DWORD *)(a2 + 8) = v33;
  *(_QWORD *)a3 = sub_2139210(a1, *(_QWORD *)a3, *(_QWORD *)(a3 + 8), a5, a6, a7);
  *(_DWORD *)(a3 + 8) = v34;
  return v34;
}
