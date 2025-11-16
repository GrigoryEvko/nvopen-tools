// Function: sub_138A5B0
// Address: 0x138a5b0
//
void __fastcall sub_138A5B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rsi
  char v6; // cl
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 *v11; // rbx
  __int64 *v12; // r14
  __int64 v13; // r13
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rsi
  __int64 v19; // r13
  __int64 v20; // rsi
  __int64 v21; // [rsp+8h] [rbp-28h]

  v3 = a2;
  switch ( *(_BYTE *)(a2 + 16) )
  {
    case 0x18:
    case 0x1A:
    case 0x1B:
    case 0x1C:
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x21:
    case 0x22:
    case 0x39:
    case 0x49:
    case 0x4A:
    case 0x4B:
    case 0x4C:
    case 0x50:
    case 0x51:
    case 0x52:
    case 0x58:
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
        goto LABEL_11;
      return;
    case 0x19:
      if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
      {
        v14 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        if ( v14 )
        {
          if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) == 15 )
          {
            sub_1389430(a1, v14, 0);
            v15 = *(_QWORD *)(a1 + 32);
            v16 = *(unsigned int *)(v15 + 8);
            if ( (unsigned int)v16 >= *(_DWORD *)(v15 + 12) )
            {
              sub_16CD150(*(_QWORD *)(a1 + 32), v15 + 16, 0, 8);
              v16 = *(unsigned int *)(v15 + 8);
            }
            *(_QWORD *)(*(_QWORD *)v15 + 8 * v16) = v14;
            ++*(_DWORD *)(v15 + 8);
          }
        }
      }
      return;
    case 0x1D:
      sub_1389CB0(a1, a2 & 0xFFFFFFFFFFFFFFFBLL);
      return;
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2A:
    case 0x2B:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    case 0x34:
      sub_13899F0(a1, a2);
      return;
    case 0x35:
      v9 = 0;
      goto LABEL_13;
    case 0x36:
    case 0x56:
      v5 = *(_QWORD *)(a2 - 24);
      v6 = 1;
      v7 = v3;
      goto LABEL_9;
    case 0x37:
      v7 = *(_QWORD *)(a2 - 24);
      v6 = 0;
      v5 = *(_QWORD *)(a2 - 48);
      goto LABEL_9;
    case 0x38:
      sub_1389870(a1, a2);
      return;
    case 0x3A:
      v7 = *(_QWORD *)(a2 - 72);
      v6 = 0;
      v5 = *(_QWORD *)(a2 - 24);
      goto LABEL_9;
    case 0x3B:
      v7 = *(_QWORD *)(a2 - 48);
      v6 = 0;
      v5 = *(_QWORD *)(a2 - 24);
      goto LABEL_9;
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
    case 0x47:
    case 0x48:
      v4 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(*(_QWORD *)v4 + 8LL) == 15 && *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
      {
        sub_1389430(a1, *(_QWORD *)(a2 - 24), 0);
        if ( v4 != a2 )
          sub_1389510(a1, v4, a2, 0);
      }
      return;
    case 0x45:
      v21 = a1;
      v3 = *(_QWORD *)(a2 - 24);
      v8 = sub_14C8190();
      goto LABEL_12;
    case 0x46:
LABEL_11:
      v21 = a1;
      v8 = sub_14C8160(a1, a2, a3);
LABEL_12:
      a1 = v21;
      v9 = v8;
      a2 = v3;
LABEL_13:
      sub_1389430(a1, a2, v9);
      return;
    case 0x4D:
      v10 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      {
        v11 = *(__int64 **)(a2 - 8);
        v12 = &v11[v10];
      }
      else
      {
        v11 = (__int64 *)(a2 - v10 * 8);
        v12 = (__int64 *)a2;
      }
      while ( v12 != v11 )
      {
        v13 = *v11;
        if ( *(_BYTE *)(*(_QWORD *)*v11 + 8LL) == 15 && *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
        {
          sub_1389430(a1, *v11, 0);
          if ( a2 != v13 )
            sub_1389510(a1, v13, a2, 0);
        }
        v11 += 3;
      }
      return;
    case 0x4E:
      sub_138A580(a1, a2);
      return;
    case 0x4F:
      v19 = *(_QWORD *)(a2 - 24);
      v20 = *(_QWORD *)(a2 - 48);
      goto LABEL_35;
    case 0x53:
      v5 = *(_QWORD *)(a2 - 48);
      v6 = 1;
      v7 = v3;
LABEL_9:
      sub_1389080(a1, v5, v7, v6);
      return;
    case 0x54:
      v17 = *(_QWORD *)(a2 - 48);
      v18 = *(_QWORD *)(a2 - 72);
      goto LABEL_33;
    case 0x55:
      v19 = *(_QWORD *)(a2 - 48);
      v20 = *(_QWORD *)(a2 - 72);
LABEL_35:
      sub_1389800(a1, v20, v3, 0);
      sub_1389800(a1, v19, v3, 0);
      break;
    case 0x57:
      v17 = *(_QWORD *)(a2 - 24);
      v18 = *(_QWORD *)(a2 - 48);
LABEL_33:
      sub_1389800(a1, v18, v3, 0);
      sub_1389080(a1, v17, v3, 0);
      break;
  }
}
