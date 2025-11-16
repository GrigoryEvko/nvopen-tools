// Function: sub_1389140
// Address: 0x1389140
//
__m128i *__fastcall sub_1389140(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __m128i *result; // rax
  __int64 v5; // rdx
  __m128i **v6; // r14
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // r14

  switch ( *(_WORD *)(a2 + 18) )
  {
    case 0xB:
    case 0xD:
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
    case 0x33:
    case 0x34:
    case 0x3D:
      v2 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v3 = *(_QWORD *)(a2 - 24 * v2);
      if ( *(_BYTE *)(*(_QWORD *)v3 + 8LL) != 15 )
        goto LABEL_7;
      result = *(__m128i **)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
      {
        sub_1389430(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), 0);
        if ( a2 != v3 )
          sub_1389510(a1, v3, a2, 0);
        v2 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
LABEL_7:
        v5 = 1;
        goto LABEL_30;
      }
      return result;
    case 0xC:
    case 0x1D:
    case 0x1E:
    case 0x1F:
    case 0x21:
    case 0x22:
    case 0x23:
    case 0x31:
    case 0x32:
    case 0x35:
    case 0x36:
    case 0x38:
    case 0x39:
    case 0x3A:
    case 0x3C:
    case 0x3F:
      v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v8 = *(_QWORD *)(a2 - 24 * v7);
      if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) != 15 )
        goto LABEL_17;
      result = *(__m128i **)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
      {
        sub_1389430(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), 0);
        if ( a2 != v8 )
          sub_1389510(a1, v8, a2, 0);
        v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
LABEL_17:
        v9 = *(_QWORD *)(a2 + 24 * (1 - v7));
        result = *(__m128i **)v9;
        if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 15 )
        {
          result = *(__m128i **)a2;
          if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
          {
            sub_1389430(a1, v9, 0);
            sub_1389430(a1, a2, 0);
            sub_13848E0(*(_QWORD *)(a1 + 24), a2, 1u, 0);
            return sub_1384420(*(_QWORD *)(a1 + 24), v9, 0, a2, 1, 0);
          }
        }
      }
      return result;
    case 0x20:
      return (__m128i *)sub_1389870();
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2A:
    case 0x2B:
    case 0x2C:
    case 0x2F:
    case 0x30:
      v6 = *(__m128i ***)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      result = *v6;
      if ( (*v6)->m128i_i8[8] == 15 )
        goto LABEL_9;
      return result;
    case 0x2D:
      v10 = sub_14C8190();
      a2 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      return (__m128i *)sub_1389430(a1, a2, v10);
    case 0x2E:
      v10 = sub_14C8160();
      return (__m128i *)sub_1389430(a1, a2, v10);
    case 0x37:
      v2 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v11 = *(_QWORD *)(a2 + 24 * (1 - v2));
      if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) != 15 )
        goto LABEL_29;
      result = *(__m128i **)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
      {
        sub_1389430(a1, v11, 0);
        if ( a2 != v11 )
          sub_1389510(a1, v11, a2, 0);
        v2 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
LABEL_29:
        v5 = 2;
LABEL_30:
        v6 = *(__m128i ***)(a2 + 24 * (v5 - v2));
        result = *v6;
        if ( (*v6)->m128i_i8[8] == 15 )
        {
LABEL_9:
          result = *(__m128i **)a2;
          if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
          {
            result = (__m128i *)sub_1389430(a1, v6, 0);
            if ( (__m128i **)a2 != v6 )
              return (__m128i *)sub_1389510(a1, v6, a2, 0);
          }
        }
      }
      return result;
    case 0x3B:
    case 0x3E:
      return sub_1389080(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), a2, 1);
  }
}
