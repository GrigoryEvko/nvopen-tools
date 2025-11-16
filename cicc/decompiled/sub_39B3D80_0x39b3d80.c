// Function: sub_39B3D80
// Address: 0x39b3d80
//
__int64 __fastcall sub_39B3D80(__int64 a1, __int64 a2, int a3)
{
  int v4; // r13d
  unsigned int v5; // ebx
  __int64 v6; // rdx
  _BYTE *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r12
  const void *v10; // r8
  const void *v11; // r9
  signed __int64 v12; // r12
  int v13; // eax
  __int64 (*v14)(); // rax
  unsigned int v15; // r14d
  const void *v17; // [rsp+0h] [rbp-90h]
  const void *v18; // [rsp+8h] [rbp-88h]
  _BYTE *v19; // [rsp+10h] [rbp-80h] BYREF
  __int64 v20; // [rsp+18h] [rbp-78h]
  _BYTE dest[112]; // [rsp+20h] [rbp-70h] BYREF

  v4 = a3;
  if ( a3 < 0 )
    v4 = *(_DWORD *)(a2 + 96);
  v5 = *(_DWORD *)(a2 + 36);
  if ( v5 )
  {
    v6 = *(_QWORD *)(a2 + 24);
    v19 = dest;
    v7 = dest;
    v8 = *(_QWORD *)(v6 + 16);
    v9 = 8LL * *(unsigned int *)(v6 + 12);
    v10 = (const void *)(v8 + 8);
    v11 = (const void *)(v8 + v9);
    v12 = v9 - 8;
    v20 = 0x800000000LL;
    v13 = 0;
    if ( (unsigned __int64)v12 > 0x40 )
    {
      v17 = v11;
      v18 = v10;
      sub_16CD150((__int64)&v19, dest, v12 >> 3, 8, (int)v10, (int)v11);
      v13 = v20;
      v11 = v17;
      v10 = v18;
      v7 = &v19[8 * (unsigned int)v20];
    }
    if ( v10 != v11 )
    {
      memcpy(v7, v10, v12);
      v13 = v20;
    }
    LODWORD(v20) = v13 + (v12 >> 3);
    if ( v5 == 33 )
    {
      v14 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 152LL);
      if ( v14 == sub_1D5A370 )
        goto LABEL_30;
    }
    else
    {
      if ( v5 != 31 )
      {
        if ( v5 <= 0x1189 )
        {
          if ( v5 <= 0x1182 )
          {
            if ( v5 > 0x95 )
            {
              if ( v5 == 191 )
LABEL_28:
                v15 = 0;
              else
                v15 = v5 != 215;
              goto LABEL_18;
            }
            if ( v5 > 2 )
            {
              switch ( v5 )
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
                  goto LABEL_28;
                default:
                  goto LABEL_17;
              }
            }
            goto LABEL_17;
          }
          v15 = ((1LL << ((unsigned __int8)v5 + 125)) & 0x49) == 0 ? 1 : 4;
LABEL_18:
          if ( v19 != dest )
            _libc_free((unsigned __int64)v19);
          return v15;
        }
LABEL_17:
        v15 = 1;
        goto LABEL_18;
      }
      v14 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 160LL);
      if ( v14 == sub_1D5A380 )
      {
LABEL_30:
        v15 = 4;
        goto LABEL_18;
      }
    }
    if ( (unsigned __int8)v14() )
      goto LABEL_17;
    goto LABEL_30;
  }
  v15 = 1;
  if ( sub_14A2090(a1, (_BYTE *)a2) )
  {
    if ( v4 < 0 )
      v4 = *(_DWORD *)(*(_QWORD *)(a2 + 24) + 12LL) - 1;
    return (unsigned int)(v4 + 1);
  }
  return v15;
}
