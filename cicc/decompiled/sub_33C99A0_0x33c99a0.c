// Function: sub_33C99A0
// Address: 0x33c99a0
//
__int64 __fastcall sub_33C99A0(__int64 a1, int a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r13
  int v7; // eax
  unsigned int v8; // eax
  unsigned int v9; // eax
  unsigned int v10; // eax
  unsigned __int64 v11; // rax
  unsigned int v12; // eax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rdx
  int v17; // eax
  unsigned int v18; // eax
  bool v19; // cc
  unsigned __int64 v20; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+8h] [rbp-38h]
  unsigned __int64 v22; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v23; // [rsp+18h] [rbp-28h]

  v4 = a3;
  switch ( a2 )
  {
    case 56:
      v23 = *(_DWORD *)(a3 + 8);
      if ( v23 > 0x40 )
        sub_C43780((__int64)&v22, (const void **)a3);
      else
        v22 = *(_QWORD *)a3;
      sub_C45EE0((__int64)&v22, a4);
      goto LABEL_7;
    case 57:
      v23 = *(_DWORD *)(a3 + 8);
      if ( v23 > 0x40 )
        sub_C43780((__int64)&v22, (const void **)a3);
      else
        v22 = *(_QWORD *)a3;
      sub_C46B40((__int64)&v22, a4);
      goto LABEL_7;
    case 58:
      sub_C472A0((__int64)&v22, a3, a4);
      goto LABEL_7;
    case 59:
      if ( sub_9867B0((__int64)a4) )
        goto LABEL_2;
      sub_C4A3E0((__int64)&v22, v4, (__int64)a4);
      goto LABEL_7;
    case 60:
      if ( sub_9867B0((__int64)a4) )
        goto LABEL_2;
      sub_C4A1D0((__int64)&v22, v4, (__int64)a4);
      goto LABEL_7;
    case 61:
      if ( sub_9867B0((__int64)a4) )
        goto LABEL_2;
      sub_C4B8A0((__int64)&v22, v4, (__int64)a4);
      goto LABEL_7;
    case 62:
      if ( sub_9867B0((__int64)a4) )
        goto LABEL_2;
      sub_C4B490((__int64)&v22, v4, (__int64)a4);
      goto LABEL_7;
    case 82:
      sub_C46090((__int64)&v22, a3, (__int64)a4);
      goto LABEL_7;
    case 83:
      sub_C49B30((__int64)&v22, a3, a4);
      goto LABEL_7;
    case 84:
      sub_C46CE0((__int64)&v22, a3, (__int64)a4);
      goto LABEL_7;
    case 85:
      sub_C49A20((__int64)&v22, a3, a4);
      goto LABEL_7;
    case 86:
      sub_C481A0((__int64)&v22, a3, (__int64)a4);
      goto LABEL_7;
    case 87:
      sub_C47DE0((__int64)&v22, a3, (__int64)a4);
      goto LABEL_7;
    case 172:
      sub_C4EBB0((__int64)&v22, a3, (const void **)a4);
      goto LABEL_7;
    case 173:
      sub_C4EAF0((__int64)&v22, (_DWORD *)a3, a4);
      goto LABEL_7;
    case 174:
      sub_C4E200((__int64)&v22, a3, a4);
      goto LABEL_7;
    case 175:
      sub_C4E440((__int64)&v22, a3, a4);
      goto LABEL_7;
    case 176:
      sub_C4E630((__int64)&v22, a3, a4);
      goto LABEL_7;
    case 177:
      sub_C4E8B0((__int64)&v22, a3, a4);
      goto LABEL_7;
    case 178:
      if ( (int)sub_C4C880(a3, (__int64)a4) < 0 )
        goto LABEL_69;
      goto LABEL_41;
    case 179:
      if ( (int)sub_C49970(a3, (unsigned __int64 *)a4) < 0 )
      {
LABEL_69:
        sub_9865C0((__int64)&v22, (__int64)a4);
        sub_C46B40((__int64)&v22, (__int64 *)v4);
        v10 = v23;
        v21 = v23;
        v20 = v22;
      }
      else
      {
LABEL_41:
        sub_9865C0((__int64)&v22, v4);
        sub_C46B40((__int64)&v22, a4);
        v9 = v23;
        v23 = 0;
        v21 = v9;
        v20 = v22;
        sub_969240((__int64 *)&v22);
        v10 = v21;
      }
      *(_DWORD *)(a1 + 8) = v10;
      if ( v10 > 0x40 )
      {
        sub_C43780(a1, (const void **)&v20);
        v19 = v21 <= 0x40;
        *(_BYTE *)(a1 + 16) = 1;
        if ( !v19 && v20 )
          j_j___libc_free_0_0(v20);
      }
      else
      {
        v11 = v20;
        *(_BYTE *)(a1 + 16) = 1;
        *(_QWORD *)a1 = v11;
      }
      return a1;
    case 180:
      v17 = sub_C4C880(a3, (__int64)a4);
      goto LABEL_64;
    case 181:
      v7 = sub_C4C880(a3, (__int64)a4);
      goto LABEL_33;
    case 182:
      v17 = sub_C49970(a3, (unsigned __int64 *)a4);
LABEL_64:
      if ( v17 > 0 )
        v4 = (__int64)a4;
      v18 = *(_DWORD *)(v4 + 8);
      *(_DWORD *)(a1 + 8) = v18;
      if ( v18 > 0x40 )
        goto LABEL_67;
      goto LABEL_36;
    case 183:
      v7 = sub_C49970(a3, (unsigned __int64 *)a4);
LABEL_33:
      if ( v7 < 0 )
        v4 = (__int64)a4;
      v8 = *(_DWORD *)(v4 + 8);
      *(_DWORD *)(a1 + 8) = v8;
      if ( v8 > 0x40 )
LABEL_67:
        sub_C43780(a1, (const void **)v4);
      else
LABEL_36:
        *(_QWORD *)a1 = *(_QWORD *)v4;
      goto LABEL_8;
    case 186:
      v12 = *(_DWORD *)(a3 + 8);
      v23 = v12;
      if ( v12 <= 0x40 )
      {
        v15 = *(_QWORD *)a3;
LABEL_57:
        v14 = *a4 & v15;
        goto LABEL_54;
      }
      sub_C43780((__int64)&v22, (const void **)a3);
      v12 = v23;
      if ( v23 <= 0x40 )
      {
        v15 = v22;
        goto LABEL_57;
      }
      sub_C43B90(&v22, a4);
      v12 = v23;
      v14 = v22;
      goto LABEL_54;
    case 187:
      v12 = *(_DWORD *)(a3 + 8);
      v23 = v12;
      if ( v12 <= 0x40 )
      {
        v13 = *(_QWORD *)a3;
LABEL_53:
        v14 = *a4 | v13;
        goto LABEL_54;
      }
      sub_C43780((__int64)&v22, (const void **)a3);
      v12 = v23;
      if ( v23 <= 0x40 )
      {
        v13 = v22;
        goto LABEL_53;
      }
      sub_C43BD0(&v22, a4);
      v12 = v23;
      v14 = v22;
      goto LABEL_54;
    case 188:
      v12 = *(_DWORD *)(a3 + 8);
      v23 = v12;
      if ( v12 <= 0x40 )
      {
        v16 = *(_QWORD *)a3;
LABEL_62:
        v14 = *a4 ^ v16;
        goto LABEL_54;
      }
      sub_C43780((__int64)&v22, (const void **)a3);
      v12 = v23;
      if ( v23 <= 0x40 )
      {
        v16 = v22;
        goto LABEL_62;
      }
      sub_C43C10(&v22, a4);
      v12 = v23;
      v14 = v22;
LABEL_54:
      *(_DWORD *)(a1 + 8) = v12;
      *(_QWORD *)a1 = v14;
      *(_BYTE *)(a1 + 16) = 1;
      return a1;
    case 190:
      v23 = *(_DWORD *)(a3 + 8);
      if ( v23 > 0x40 )
        sub_C43780((__int64)&v22, (const void **)a3);
      else
        v22 = *(_QWORD *)a3;
      sub_C47AC0((__int64)&v22, (__int64)a4);
      goto LABEL_7;
    case 191:
      v23 = *(_DWORD *)(a3 + 8);
      if ( v23 > 0x40 )
        sub_C43780((__int64)&v22, (const void **)a3);
      else
        v22 = *(_QWORD *)a3;
      sub_C44D10((__int64)&v22, (unsigned __int64)a4);
      goto LABEL_7;
    case 192:
      v23 = *(_DWORD *)(a3 + 8);
      if ( v23 > 0x40 )
        sub_C43780((__int64)&v22, (const void **)a3);
      else
        v22 = *(_QWORD *)a3;
      sub_C48380((__int64)&v22, (__int64)a4);
      goto LABEL_7;
    case 193:
      sub_C4B840((__int64)&v22, a3, (__int64)a4);
      goto LABEL_7;
    case 194:
      sub_C4B870((__int64)&v22, a3, (__int64)a4);
LABEL_7:
      *(_DWORD *)(a1 + 8) = v23;
      *(_QWORD *)a1 = v22;
LABEL_8:
      *(_BYTE *)(a1 + 16) = 1;
      return a1;
    default:
LABEL_2:
      *(_BYTE *)(a1 + 16) = 0;
      return a1;
  }
}
