// Function: sub_1B93750
// Address: 0x1b93750
//
__int64 __fastcall sub_1B93750(__int64 a1, __int64 a2, __int64 a3, int *a4)
{
  __int64 v7; // rdi
  unsigned int v8; // eax
  unsigned int v9; // r12d
  __int64 v10; // rdx
  int v11; // eax
  unsigned __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rcx
  __int64 v20; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v21[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 (__fastcall *v22)(const __m128i **, const __m128i *, int); // [rsp+20h] [rbp-40h]
  __int64 (__fastcall *v23)(); // [rsp+28h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 32);
  v20 = a2;
  v8 = sub_1B91FD0(v7, a2);
  if ( (_BYTE)v8 )
  {
    return 0;
  }
  else
  {
    v9 = v8;
    v10 = *(unsigned __int8 *)(v20 + 16);
    switch ( *(_BYTE *)(v20 + 16) )
    {
      case 0x1A:
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
      case 0x36:
      case 0x37:
      case 0x38:
      case 0x3C:
      case 0x3D:
      case 0x3E:
      case 0x3F:
      case 0x40:
      case 0x41:
      case 0x42:
      case 0x43:
      case 0x44:
      case 0x45:
      case 0x46:
      case 0x47:
      case 0x4B:
      case 0x4C:
      case 0x4D:
      case 0x4E:
      case 0x4F:
        if ( (_BYTE)v10 == 78 )
        {
          v11 = sub_14C3B40(v20, *(__int64 **)(a1 + 8));
          if ( v11 )
          {
            v10 = (unsigned int)(v11 - 116);
            if ( (unsigned int)v10 <= 1 || v11 == 4 || v11 == 191 )
              return 0;
          }
        }
        v21[1] = a1;
        v21[0] = &v20;
        v23 = sub_1B97BE0;
        v22 = sub_1B8E250;
        v9 = sub_1B932A0((__int64)v21, a4, v10);
        if ( v22 )
          v22((const __m128i **)v21, (const __m128i *)v21, 3);
        if ( !(_BYTE)v9 )
          return 0;
        v13 = *(_QWORD *)(a3 + 112) & 0xFFFFFFFFFFFFFFF8LL;
        if ( a3 + 112 == v13 )
          goto LABEL_17;
        if ( !v13 )
          BUG();
        if ( *(_BYTE *)(v13 + 16) == 9 && (v14 = *(_QWORD *)(v13 + 40), v14 == v20 + 24) )
        {
          *(_QWORD *)(v13 + 40) = *(_QWORD *)(v14 + 8);
        }
        else
        {
LABEL_17:
          v15 = sub_22077B0(56);
          if ( v15 )
          {
            v16 = v20;
            *(_QWORD *)(v15 + 40) = 0;
            v17 = 0;
            *(_QWORD *)(v15 + 48) = 0;
            *(_QWORD *)(v15 + 8) = 0;
            v16 += 24;
            *(_QWORD *)(v15 + 16) = 0;
            *(_BYTE *)(v15 + 24) = 9;
            *(_QWORD *)(v15 + 32) = 0;
            *(_QWORD *)v15 = &unk_49F6EB8;
            v18 = *(_QWORD *)(v16 + 8);
            *(_QWORD *)(v15 + 40) = v16;
            *(_QWORD *)(v15 + 48) = v18;
          }
          else
          {
            v17 = MEMORY[8] & 7;
          }
          v19 = *(_QWORD *)(a3 + 112);
          *(_QWORD *)(v15 + 32) = a3;
          *(_QWORD *)(v15 + 16) = a3 + 112;
          v19 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v15 + 8) = v19 | v17;
          *(_QWORD *)(v19 + 8) = v15 + 8;
          *(_QWORD *)(a3 + 112) = *(_QWORD *)(a3 + 112) & 7LL | (v15 + 8);
        }
        break;
      default:
        return v9;
    }
  }
  return v9;
}
