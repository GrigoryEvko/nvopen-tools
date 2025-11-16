// Function: sub_21686A0
// Address: 0x21686a0
//
__int64 __fastcall sub_21686A0(__int64 a1, _BYTE *a2, int a3, __int64 a4)
{
  int v5; // r13d
  unsigned int v6; // ebx
  unsigned __int64 v7; // rdx
  _BYTE *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r12
  _BYTE *v11; // r8
  _BYTE *v12; // r9
  signed __int64 v13; // r12
  int v14; // eax
  __int64 v15; // rdi
  __int64 (*v16)(); // rax
  unsigned int v17; // r14d
  __int64 v19; // rdi
  __int64 (*v20)(); // rax
  _BYTE *v21; // [rsp+0h] [rbp-90h]
  _BYTE *v22; // [rsp+8h] [rbp-88h]
  _BYTE *v23; // [rsp+10h] [rbp-80h] BYREF
  __int64 v24; // [rsp+18h] [rbp-78h]
  _BYTE dest[112]; // [rsp+20h] [rbp-70h] BYREF

  v5 = a3;
  if ( a3 < 0 )
    v5 = *((_DWORD *)a2 + 24);
  v6 = *((_DWORD *)a2 + 9);
  if ( v6 )
  {
    v7 = *((_QWORD *)a2 + 3);
    v23 = dest;
    v8 = dest;
    v9 = *(_QWORD *)(v7 + 16);
    v10 = 8LL * *(unsigned int *)(v7 + 12);
    v11 = (_BYTE *)(v9 + 8);
    v12 = (_BYTE *)(v9 + v10);
    v13 = v10 - 8;
    v24 = 0x800000000LL;
    v14 = 0;
    if ( (unsigned __int64)v13 > 0x40 )
    {
      a2 = dest;
      v21 = v12;
      v22 = v11;
      sub_16CD150((__int64)&v23, dest, v13 >> 3, 8, (int)v11, (int)v12);
      v14 = v24;
      v7 = (unsigned __int64)v23;
      v12 = v21;
      v11 = v22;
      v8 = &v23[8 * (unsigned int)v24];
    }
    if ( v11 != v12 )
    {
      a2 = v11;
      memcpy(v8, v11, v13);
      v14 = v24;
    }
    LODWORD(v24) = v14 + (v13 >> 3);
    if ( v6 == 33 )
    {
      v19 = *(_QWORD *)(a1 + 16);
      v20 = *(__int64 (**)())(*(_QWORD *)v19 + 152LL);
      if ( v20 != sub_1D5A370
        && ((unsigned __int8 (__fastcall *)(__int64, _BYTE *, unsigned __int64, __int64, _BYTE *, _BYTE *))v20)(
             v19,
             a2,
             v7,
             a4,
             v11,
             v12) )
      {
        goto LABEL_16;
      }
    }
    else
    {
      if ( v6 != 31 )
      {
        if ( v6 <= 0x1189 )
        {
          if ( v6 <= 0x1182 )
          {
            if ( v6 > 0x95 )
            {
              if ( v6 == 191 )
LABEL_27:
                v17 = 0;
              else
                v17 = v6 != 215;
              goto LABEL_17;
            }
            if ( v6 > 2 )
            {
              switch ( v6 )
              {
                case 3u:
                case 4u:
                case 0xEu:
                case 0xFu:
                case 0x12u:
                case 0x13u:
                case 0x14u:
                case 0x17u:
                case 0x1Bu:
                case 0x1Cu:
                case 0x1Du:
                case 0x24u:
                case 0x25u:
                case 0x26u:
                case 0x4Cu:
                case 0x4Du:
                case 0x71u:
                case 0x72u:
                case 0x74u:
                case 0x75u:
                case 0x90u:
                case 0x95u:
                  goto LABEL_27;
                default:
                  goto LABEL_16;
              }
            }
            goto LABEL_16;
          }
          v17 = ((1LL << ((unsigned __int8)v6 + 125)) & 0x49) == 0 ? 1 : 4;
LABEL_17:
          if ( v23 != dest )
            _libc_free((unsigned __int64)v23);
          return v17;
        }
LABEL_16:
        v17 = 1;
        goto LABEL_17;
      }
      v15 = *(_QWORD *)(a1 + 16);
      v16 = *(__int64 (**)())(*(_QWORD *)v15 + 160LL);
      if ( v16 == sub_2165300
        || ((unsigned __int8 (__fastcall *)(__int64, _BYTE *, unsigned __int64, __int64, _BYTE *, _BYTE *))v16)(
             v15,
             a2,
             v7,
             a4,
             v11,
             v12) )
      {
        goto LABEL_16;
      }
    }
    v17 = 4;
    goto LABEL_17;
  }
  v17 = 1;
  if ( sub_14A2090(a1, a2) )
  {
    if ( v5 < 0 )
      v5 = *(_DWORD *)(*((_QWORD *)a2 + 3) + 12LL) - 1;
    return (unsigned int)(v5 + 1);
  }
  return v17;
}
