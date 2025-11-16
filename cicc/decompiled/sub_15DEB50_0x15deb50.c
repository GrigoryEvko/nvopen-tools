// Function: sub_15DEB50
// Address: 0x15deb50
//
__int64 __fastcall sub_15DEB50(unsigned int *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r9
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned int v9; // edx
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // r8
  int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // ecx
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned __int64 v38; // [rsp+8h] [rbp-38h]

  v4 = 0x2000000000ALL;
  while ( 2 )
  {
    v7 = *a1;
    v8 = (unsigned int)(v7 + 1);
    v9 = *a1;
    *a1 = v8;
    switch ( *(_BYTE *)(a2 + v7) )
    {
      case 0:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 0;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 1:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 0x100000009LL;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 2:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 0x800000009LL;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 3:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 0x1000000009LL;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 4:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 0x2000000009LL;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 5:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 0x4000000009LL;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 6:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 5;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 7:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 6;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 8:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 7;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 9:
        v28 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v28 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          v28 = *(unsigned int *)(a4 + 8);
          v4 = 0x2000000000ALL;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v28) = 0x20000000ALL;
        ++*(_DWORD *)(a4 + 8);
        continue;
      case 0xA:
        v27 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v27 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          v27 = *(unsigned int *)(a4 + 8);
          v4 = 0x2000000000ALL;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v27) = 0x40000000ALL;
        ++*(_DWORD *)(a4 + 8);
        continue;
      case 0xB:
        v36 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v36 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          v36 = *(unsigned int *)(a4 + 8);
          v4 = 0x2000000000ALL;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v36) = 0x80000000ALL;
        ++*(_DWORD *)(a4 + 8);
        continue;
      case 0xC:
        v31 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v31 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          v31 = *(unsigned int *)(a4 + 8);
          v4 = 0x2000000000ALL;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v31) = 0x100000000ALL;
        ++*(_DWORD *)(a4 + 8);
        continue;
      case 0xD:
        v32 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v32 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          v32 = *(unsigned int *)(a4 + 8);
          v4 = 0x2000000000ALL;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v32) = 0x200000000ALL;
        ++*(_DWORD *)(a4 + 8);
        continue;
      case 0xE:
        v25 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v25 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          v25 = *(unsigned int *)(a4 + 8);
          v4 = 0x2000000000ALL;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v25) = 11;
        ++*(_DWORD *)(a4 + 8);
        continue;
      case 0xF:
        v37 = 0;
        if ( v8 != a3 )
        {
          *a1 = v9 + 2;
          v37 = *(unsigned __int8 *)(a2 + v8);
        }
        v19 = (v37 << 32) | 0xD;
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
          goto LABEL_31;
        goto LABEL_24;
      case 0x10:
        v33 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v33 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          v33 = *(unsigned int *)(a4 + 8);
          v4 = 0x2000000000ALL;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v33) = 0x400000000ALL;
        ++*(_DWORD *)(a4 + 8);
        continue;
      case 0x11:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 2;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 0x12:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 3;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 0x13:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 4;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 0x14:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 12;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 0x15:
        LODWORD(v38) = 2;
        goto LABEL_7;
      case 0x16:
        LODWORD(v38) = 3;
        goto LABEL_7;
      case 0x17:
        LODWORD(v38) = 4;
        goto LABEL_7;
      case 0x18:
        LODWORD(v38) = 5;
        goto LABEL_7;
      case 0x19:
        v35 = 0;
        if ( v8 != a3 )
        {
          *a1 = v9 + 2;
          v35 = *(unsigned __int8 *)(a2 + v8);
        }
        v19 = (v35 << 32) | 0xE;
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
          goto LABEL_31;
        goto LABEL_24;
      case 0x1A:
        v26 = 0;
        if ( v8 != a3 )
        {
          *a1 = v9 + 2;
          v26 = *(unsigned __int8 *)(a2 + v8);
        }
        v19 = (v26 << 32) | 0xF;
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
          goto LABEL_31;
        goto LABEL_24;
      case 0x1B:
        *a1 = v9 + 2;
        v29 = *(unsigned int *)(a4 + 8);
        v30 = ((unsigned __int64)*(unsigned __int8 *)(a2 + v8) << 32) | 0xB;
        if ( (unsigned int)v29 >= *(_DWORD *)(a4 + 12) )
        {
          v38 = ((unsigned __int64)*(unsigned __int8 *)(a2 + v8) << 32) | 0xB;
          sub_16CD150(a4, a4 + 16, 0, 8);
          v29 = *(unsigned int *)(a4 + 8);
          v30 = v38;
          v4 = 0x2000000000ALL;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v29) = v30;
        ++*(_DWORD *)(a4 + 8);
        continue;
      case 0x1C:
        v24 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v24 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          v24 = *(unsigned int *)(a4 + 8);
          v4 = 0x2000000000ALL;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v24) = 0x10000000ALL;
        ++*(_DWORD *)(a4 + 8);
        continue;
      case 0x1D:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 1;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 0x1E:
        v23 = 0;
        if ( v8 != a3 )
        {
          *a1 = v9 + 2;
          v23 = *(unsigned __int8 *)(a2 + v8);
        }
        v19 = (v23 << 32) | 0x10;
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
          goto LABEL_31;
        goto LABEL_24;
      case 0x1F:
        v22 = 0;
        if ( v8 != a3 )
        {
          *a1 = v9 + 2;
          v22 = *(unsigned __int8 *)(a2 + v8);
        }
        v19 = (v22 << 32) | 0x11;
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
          goto LABEL_31;
        goto LABEL_24;
      case 0x20:
        v20 = 0;
        if ( v8 != a3 )
        {
          *a1 = v9 + 2;
          v20 = *(unsigned __int8 *)(a2 + v8);
        }
        v19 = (v20 << 32) | 0x12;
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
          goto LABEL_31;
        goto LABEL_24;
      case 0x21:
        v21 = 0;
        if ( v8 != a3 )
        {
          *a1 = v9 + 2;
          v21 = *(unsigned __int8 *)(a2 + v8);
        }
        v19 = (v21 << 32) | 0x13;
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
          goto LABEL_31;
        goto LABEL_24;
      case 0x22:
        v16 = 0;
        if ( v8 != a3 )
        {
          v17 = v9 + 2;
          *a1 = v17;
          v18 = *(unsigned __int8 *)(a2 + v8);
          if ( v17 == a3 )
          {
            v16 = (unsigned __int8)v18 << 16;
          }
          else
          {
            *a1 = v9 + 3;
            v16 = (v18 << 16) | (unsigned int)*(unsigned __int8 *)(a2 + v17);
          }
        }
        v19 = (v16 << 32) | 0x14;
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result < *(_DWORD *)(a4 + 12) )
          goto LABEL_24;
LABEL_31:
        sub_16CD150(a4, a4 + 16, 0, 8);
        result = *(unsigned int *)(a4 + 8);
LABEL_24:
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = v19;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 0x23:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 0x8000000009LL;
        ++*(_DWORD *)(a4 + 8);
        return result;
      case 0x24:
        v15 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v15 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          v15 = *(unsigned int *)(a4 + 8);
          v4 = 0x2000000000ALL;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v15) = 0x2000000000ALL;
        ++*(_DWORD *)(a4 + 8);
        continue;
      case 0x25:
        v14 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v14 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          v14 = *(unsigned int *)(a4 + 8);
          v4 = 0x2000000000ALL;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v14) = 0x4000000000ALL;
        ++*(_DWORD *)(a4 + 8);
        continue;
      case 0x26:
        LODWORD(v38) = 6;
        goto LABEL_7;
      case 0x27:
        LODWORD(v38) = 7;
        goto LABEL_7;
      case 0x28:
        LODWORD(v38) = 8;
LABEL_7:
        v11 = *(unsigned int *)(a4 + 8);
        v12 = (v38 << 32) | 0xC;
        if ( (unsigned int)v11 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          v11 = *(unsigned int *)(a4 + 8);
          v12 = (v38 << 32) | 0xC;
        }
        v13 = 0;
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v11) = v12;
        ++*(_DWORD *)(a4 + 8);
        do
        {
          result = sub_15DEB50(a1, a2, a3, a4, v12, v4);
          ++v13;
        }
        while ( (_DWORD)v38 != v13 );
        break;
      case 0x29:
        result = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          result = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 8;
        ++*(_DWORD *)(a4 + 8);
        break;
      case 0x2A:
        v34 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v34 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          v34 = *(unsigned int *)(a4 + 8);
          v4 = 0x2000000000ALL;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v34) = 0x800000000ALL;
        ++*(_DWORD *)(a4 + 8);
        continue;
    }
    return result;
  }
}
