// Function: sub_26A9950
// Address: 0x26a9950
//
__int64 __fastcall sub_26A9950(__int64 a1, _BYTE *a2, unsigned int a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned int v10; // ecx
  __int64 v11; // rdx
  _BYTE *v12; // r9
  __int64 result; // rax
  char v14; // dl
  __int64 v15; // rdx
  char v16; // dl
  char v17; // dl
  __int64 v18; // rdi
  char v19; // al
  int v20; // edx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdi
  char v24; // cl
  __int64 v25; // rdi
  char v26; // al
  _QWORD *v27; // rax
  int v28; // r10d
  __int64 v29[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = *(_QWORD *)a1;
  v6 = *(_QWORD *)(*(_QWORD *)a1 + 208LL);
  v7 = *(_QWORD *)(v6 + 34560);
  v8 = *(unsigned int *)(v6 + 34576);
  if ( !(_DWORD)v8 )
    goto LABEL_18;
  v10 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = v7 + 16LL * v10;
  v12 = *(_BYTE **)v11;
  if ( a2 != *(_BYTE **)v11 )
  {
    v20 = 1;
    while ( v12 != (_BYTE *)-4096LL )
    {
      v28 = v20 + 1;
      v10 = (v8 - 1) & (v20 + v10);
      v11 = v7 + 16LL * v10;
      v12 = *(_BYTE **)v11;
      if ( a2 == *(_BYTE **)v11 )
        goto LABEL_3;
      v20 = v28;
    }
LABEL_18:
    if ( a2 )
    {
      if ( !sub_B2FC80((__int64)a2) )
      {
        result = sub_B2FC00(a2);
        if ( !(_BYTE)result )
          return result;
      }
      v21 = *(_QWORD *)(v5 + 208);
      if ( *(_BYTE *)(v21 + 276) )
      {
        result = *(_QWORD *)(v21 + 256);
        v22 = result + 8LL * *(unsigned int *)(v21 + 268);
        if ( result != v22 )
        {
          while ( a2 != *(_BYTE **)result )
          {
            result += 8;
            if ( v22 == result )
              goto LABEL_24;
          }
          return result;
        }
      }
      else
      {
        result = (__int64)sub_C8CA60(v21 + 248, (__int64)a2);
        if ( result )
          return result;
      }
LABEL_24:
      if ( *(_QWORD *)(v5 + 4432) )
      {
        result = (*(__int64 (__fastcall **)(__int64, _BYTE *))(v5 + 4440))(v5 + 4416, a2);
        if ( (_BYTE)result )
          return result;
      }
    }
    v23 = **(_QWORD **)(a1 + 8);
    if ( !v23
      || !(*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v23 + 112LL))(
            v23,
            "omp_no_openmp",
            13)
      && !(*(unsigned __int8 (__fastcall **)(_QWORD, const char *, __int64))(***(_QWORD ***)(a1 + 8) + 112LL))(
            **(_QWORD **)(a1 + 8),
            "omp_no_parallelism",
            18) )
    {
      v25 = *(_QWORD *)(a1 + 16);
      v29[0] = *(_QWORD *)(a1 + 24);
      v26 = *(_BYTE *)(v25 + 176);
      v25 += 184;
      *(_BYTE *)(v25 - 7) = v26;
      sub_269BFF0(v25, v29);
    }
    result = *(_QWORD *)(a1 + 16);
    v17 = *(_BYTE *)(result + 240);
    if ( *(_BYTE *)(result + 241) != v17 )
    {
      *(_BYTE *)(result + 241) = v17;
      v29[0] = *(_QWORD *)(a1 + 24);
      sub_269CCD0(*(_QWORD *)(a1 + 16) + 248LL, v29);
      result = *(_QWORD *)(a1 + 16);
      v17 = *(_BYTE *)(result + 241);
    }
    v24 = *(_BYTE *)(result + 401);
    *(_BYTE *)(result + 96) = 1;
    *(_BYTE *)(result + 400) = v24;
    *(_BYTE *)(result + 336) = *(_BYTE *)(result + 337);
    goto LABEL_12;
  }
LABEL_3:
  if ( v11 == v7 + 16 * v8 )
    goto LABEL_18;
  if ( a3 <= 1 )
  {
    result = *(unsigned int *)(v11 + 8);
    switch ( *(_DWORD *)(v11 + 8) )
    {
      case 0:
      case 3:
      case 4:
      case 5:
      case 6:
      case 0xE:
      case 0xF:
      case 0x10:
      case 0x11:
      case 0x12:
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
      case 0x17:
      case 0x18:
      case 0x19:
      case 0x1A:
      case 0x1B:
      case 0x1C:
      case 0x1D:
      case 0x1E:
      case 0x1F:
      case 0x20:
      case 0x21:
      case 0x22:
      case 0x23:
      case 0x24:
      case 0x25:
      case 0x26:
      case 0x27:
      case 0x28:
      case 0x2E:
      case 0x2F:
      case 0x41:
      case 0x46:
      case 0x60:
      case 0x61:
      case 0xB0:
      case 0xB1:
      case 0xBA:
        goto LABEL_11;
      case 0x3D:
      case 0x3E:
      case 0x3F:
      case 0x40:
      case 0x42:
      case 0x43:
      case 0x44:
      case 0x45:
        v15 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 32 * (2LL - (*(_DWORD *)(*(_QWORD *)(a1 + 24) + 4LL) & 0x7FFFFFF)));
        if ( *(_BYTE *)v15 != 17 )
          goto LABEL_10;
        v27 = *(_QWORD **)(v15 + 24);
        if ( *(_DWORD *)(v15 + 32) > 0x40u )
          v27 = (_QWORD *)*v27;
        if ( (unsigned int)v27 > 0x22 )
        {
          if ( (unsigned int)((_DWORD)v27 - 91) <= 1 )
          {
LABEL_11:
            result = *(_QWORD *)(a1 + 16);
            v16 = *(_BYTE *)(result + 401);
            *(_BYTE *)(result + 96) = 1;
            *(_BYTE *)(result + 400) = v16;
            *(_BYTE *)(result + 336) = *(_BYTE *)(result + 337);
            v17 = *(_BYTE *)(result + 241);
LABEL_12:
            *(_BYTE *)(result + 240) = v17;
            *(_BYTE *)(result + 112) = *(_BYTE *)(result + 113);
            *(_BYTE *)(result + 176) = *(_BYTE *)(result + 177);
            return result;
          }
        }
        else if ( (unsigned int)v27 > 0x20 )
        {
          goto LABEL_11;
        }
LABEL_10:
        *(_BYTE *)(*(_QWORD *)(a1 + 16) + 241LL) = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 240LL);
        v29[0] = *(_QWORD *)(a1 + 24);
        sub_269CCD0(*(_QWORD *)(a1 + 16) + 248LL, v29);
        goto LABEL_11;
      case 0x63:
        *(_BYTE *)(*(_QWORD *)(a1 + 16) + 241LL) = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 240LL);
        v29[0] = *(_QWORD *)(a1 + 24);
        sub_269CCD0(*(_QWORD *)(a1 + 16) + 248LL, v29);
        v18 = *(_QWORD *)(a1 + 16);
        v29[0] = *(_QWORD *)(a1 + 24);
        v19 = *(_BYTE *)(v18 + 176);
        v18 += 184;
        *(_BYTE *)(v18 - 7) = v19;
        sub_269BFF0(v18, v29);
        goto LABEL_11;
      case 0x9B:
        *(_QWORD *)(*(_QWORD *)(a1 + 16) + 296LL) = *(_QWORD *)(a1 + 24);
        goto LABEL_11;
      case 0x9C:
        *(_QWORD *)(*(_QWORD *)(a1 + 16) + 312LL) = *(_QWORD *)(a1 + 24);
        goto LABEL_11;
      case 0x9E:
        result = sub_26A95D0(*(_QWORD *)(a1 + 16), v5, *(_QWORD *)(a1 + 24));
        if ( !(_BYTE)result )
          break;
        return result;
      case 0xB4:
      case 0xB5:
        return result;
      default:
        goto LABEL_10;
    }
  }
  result = *(_QWORD *)(a1 + 16);
  v14 = *(_BYTE *)(result + 400);
  *(_BYTE *)(result + 96) = 1;
  *(_BYTE *)(result + 464) = 1;
  *(_BYTE *)(result + 401) = v14;
  *(_BYTE *)(result + 337) = *(_BYTE *)(result + 336);
  *(_BYTE *)(result + 241) = *(_BYTE *)(result + 240);
  *(_BYTE *)(result + 113) = *(_BYTE *)(result + 112);
  *(_BYTE *)(result + 177) = *(_BYTE *)(result + 176);
  return result;
}
