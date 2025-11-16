// Function: sub_7D9310
// Address: 0x7d9310
//
__int64 __fastcall sub_7D9310(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // kr00_8
  __int64 result; // rax
  __int64 v10; // r13
  _BYTE *v11; // rax
  _QWORD *v12; // rax
  _BYTE *v13; // r14
  _BYTE *v14; // rax
  _BYTE *v15; // rax
  __int64 *v16; // rcx
  __int64 v17; // r15
  __int64 *v18; // r8
  unsigned __int8 v19; // al
  void *v20; // rbx
  __int64 *v21; // rsi
  _BYTE *v22; // rbx
  __int64 v23; // rbx
  __int64 v24; // r14
  __int64 i; // rax
  __int64 v26; // r13
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rsi
  _QWORD *v30; // r15
  __int64 *v31; // r12
  bool v32; // zf
  __int64 v33; // r14
  __int64 v34; // rcx
  __int64 *v35; // rax
  __int64 *v36; // rax
  __int64 v37; // [rsp-48h] [rbp-48h]
  __int64 *v38; // [rsp-40h] [rbp-40h]
  __int64 v39; // [rsp-40h] [rbp-40h]
  __int64 v40; // [rsp-40h] [rbp-40h]
  __int64 *v41; // [rsp-40h] [rbp-40h]
  __int64 *v42; // [rsp-40h] [rbp-40h]
  __int64 *v43; // [rsp-40h] [rbp-40h]

  v7 = *((unsigned __int8 *)a1 + 57);
  v8 = v6;
  result = *((unsigned __int8 *)a1 + 56);
  switch ( *((_BYTE *)a1 + 56) )
  {
    case 0:
    case 0x15:
      return sub_7313A0(a1[9], a2, a3, v7, a5, a6);
    case 1:
    case 2:
    case 3:
    case 4:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
    case 0xE:
    case 0xF:
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13:
    case 0x16:
    case 0x17:
    case 0x18:
    case 0x19:
    case 0x1C:
    case 0x1D:
    case 0x1E:
    case 0x1F:
    case 0x2B:
    case 0x32:
    case 0x33:
    case 0x34:
    case 0x35:
    case 0x36:
    case 0x37:
    case 0x38:
    case 0x39:
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
    case 0x48:
    case 0x49:
    case 0x54:
    case 0x55:
    case 0x56:
    case 0x57:
    case 0x58:
    case 0x59:
    case 0x5A:
    case 0x5C:
    case 0x5D:
    case 0x5E:
    case 0x5F:
    case 0x60:
    case 0x61:
    case 0x62:
    case 0x63:
    case 0x64:
    case 0x65:
    case 0x66:
    case 0x67:
    case 0x68:
      return result;
    case 5:
    case 0x14:
      return (__int64)sub_7D8AA0(a1);
    case 0x1A:
      if ( (_BYTE)v7 == 5 )
        return sub_7D7D40(a1);
      return result;
    case 0x1B:
      return sub_730620((__int64)a1, (const __m128i *)a1[9]);
    case 0x20:
      return sub_7D8330(a1);
    case 0x21:
    case 0x22:
      return sub_7D8400((__int64)a1);
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
      if ( (_BYTE)v7 == 2 )
      {
        result = sub_8D29A0(*a1);
        if ( (_DWORD)result )
          return sub_7F0D20(a1);
      }
      else if ( (_BYTE)v7 == 5 )
      {
        return (__int64)sub_7D8FC0((__int64)a1);
      }
      return result;
    case 0x27:
      if ( (_BYTE)v7 == 5 )
        return sub_7D7E10(a1);
      return result;
    case 0x28:
      if ( (_BYTE)v7 == 5 )
        return sub_7D7EE0(a1);
      return result;
    case 0x29:
      if ( (_BYTE)v7 == 5 )
        return sub_7D7FB0(a1);
      return result;
    case 0x2A:
      if ( (_BYTE)v7 == 5 )
        return sub_7D8080(a1);
      return result;
    case 0x2C:
      result = (__int64)sub_73DBF0(0x29u, *a1, a1[9]);
      *((_BYTE *)a1 + 56) = 26;
      a1[9] = result;
      return result;
    case 0x2D:
      result = (__int64)sub_73DBF0(0x2Au, *a1, a1[9]);
      *((_BYTE *)a1 + 56) = 26;
      a1[9] = result;
      return result;
    case 0x2E:
    case 0x2F:
    case 0x30:
    case 0x31:
      v10 = sub_7E7CA0(*a1);
      v11 = sub_731250(v10);
      v12 = (_QWORD *)sub_7D7B30((__int64)v11);
      v13 = sub_73DCD0(v12);
      v14 = sub_731250(v10);
      v15 = sub_7D7C20((__int64)v14);
      v16 = (__int64 *)a1[9];
      v17 = (__int64)v15;
      v18 = (__int64 *)v16[2];
      v16[2] = 0;
      v19 = *((_BYTE *)a1 + 56);
      if ( v19 == 48 )
      {
        *((_QWORD *)v13 + 2) = v16;
        v43 = v18;
        v20 = sub_73DBF0(0x49u, *v16, (__int64)v13);
        v36 = (__int64 *)sub_73DBF0(0x1Au, *v43, (__int64)v43);
        *(_QWORD *)(v17 + 16) = v36;
        v21 = (__int64 *)sub_73DBF0(0x49u, *v36, v17);
        goto LABEL_12;
      }
      if ( v19 > 0x30u )
      {
        if ( v19 == 49 )
        {
          *(_QWORD *)(v17 + 16) = v16;
          v41 = v18;
          v20 = sub_73DBF0(0x49u, *v16, v17);
          v35 = (__int64 *)sub_73DBF0(0x1Au, *v41, (__int64)v41);
          *((_QWORD *)v13 + 2) = v35;
          v21 = (__int64 *)sub_73DBF0(0x49u, *v35, (__int64)v13);
          goto LABEL_12;
        }
      }
      else
      {
        if ( v19 == 46 )
        {
          *((_QWORD *)v13 + 2) = v16;
          v38 = v18;
          v20 = sub_73DBF0(0x49u, *v16, (__int64)v13);
          *(_QWORD *)(v17 + 16) = v38;
          v21 = (__int64 *)sub_73DBF0(0x49u, *v38, v17);
LABEL_12:
          v22 = sub_73DF90((__int64)v20, v21);
          *((_QWORD *)v22 + 2) = sub_73E830(v10);
          return sub_73D8E0((__int64)a1, 0x5Bu, *a1, 0, (__int64)v22);
        }
        if ( v19 == 47 )
        {
          *(_QWORD *)(v17 + 16) = v16;
          v42 = v18;
          v20 = sub_73DBF0(0x49u, *v16, v17);
          *((_QWORD *)v13 + 2) = v42;
          v21 = (__int64 *)sub_73DBF0(0x49u, *v42, (__int64)v13);
          goto LABEL_12;
        }
      }
      sub_721090();
    case 0x3A:
      if ( (_BYTE)v7 == 5 )
        return sub_7D8150(a1);
      return result;
    case 0x3B:
      if ( (_BYTE)v7 == 5 )
        return sub_7D8240(a1);
      return result;
    case 0x4A:
    case 0x4B:
    case 0x4C:
    case 0x4D:
    case 0x4E:
    case 0x4F:
    case 0x50:
    case 0x51:
    case 0x52:
    case 0x53:
      if ( (unsigned __int8)v7 > 3u )
      {
        if ( (unsigned __int8)(v7 - 4) <= 1u )
          return sub_7F12E0(a1);
      }
      else if ( (unsigned __int8)v7 > 1u )
      {
        result = sub_8D29A0(*a1);
        if ( (_DWORD)result )
          return sub_7F12E0(a1);
      }
      return result;
    case 0x5B:
      return sub_7E6FC0(a1, a2, a3, v7);
    case 0x69:
      v23 = a1[9];
      result = sub_72B0F0(v23, 0);
      v24 = result;
      if ( !result )
        return result;
      for ( i = *(_QWORD *)(result + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v26 = *(_QWORD *)(i + 168);
      for ( result = sub_8D46C0(*(_QWORD *)v23); *(_BYTE *)(result + 140) == 12; result = *(_QWORD *)(result + 160) )
        ;
      if ( (*(_BYTE *)(v26 + 16) & 2) == 0 )
        return result;
      if ( (*(_BYTE *)(*(_QWORD *)(result + 168) + 16LL) & 2) == 0 )
        return result;
      v29 = *(_QWORD *)(v24 + 152);
      if ( v29 == result )
        return result;
      result = sub_8D97D0(result, v29, 0, v27, v28);
      if ( (_DWORD)result )
        return result;
      v30 = *(_QWORD **)v26;
      v31 = (__int64 *)(v23 + 16);
      v39 = *(_QWORD *)(a1[9] + 16);
      v37 = *(_QWORD *)v23;
      result = sub_72D2E0(*(_QWORD **)(v24 + 152));
      v32 = *(_BYTE *)(v23 + 24) == 20;
      *(_QWORD *)v23 = result;
      if ( !v32 )
      {
        result = *(_QWORD *)(v23 + 72);
        *(_QWORD *)result = *(_QWORD *)(v24 + 152);
      }
      if ( !v39 )
        goto LABEL_66;
      v33 = *(_QWORD *)(v39 + 16);
      if ( !v30 )
        goto LABEL_35;
      do
      {
        v34 = *(_QWORD *)(*v31 + 16);
        *(_QWORD *)(*v31 + 16) = 0;
        v40 = v34;
        result = (__int64)sub_73E130((_QWORD *)*v31, v30[1]);
        *v31 = result;
        *(_QWORD *)(result + 16) = v40;
        v30 = (_QWORD *)*v30;
        v31 = (__int64 *)(*v31 + 16);
        if ( !v33 )
        {
LABEL_66:
          if ( !v30 )
            return result;
          goto LABEL_67;
        }
        v33 = *(_QWORD *)(v33 + 16);
      }
      while ( v30 );
LABEL_35:
      if ( (*(_BYTE *)(v26 + 16) & 1) == 0 )
      {
LABEL_67:
        *(_QWORD *)v23 = v37;
        return v37;
      }
      while ( v33 )
        v33 = *(_QWORD *)(v33 + 16);
      return result;
    default:
      return v8;
  }
}
