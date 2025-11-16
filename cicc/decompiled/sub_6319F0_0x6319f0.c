// Function: sub_6319F0
// Address: 0x6319f0
//
__int64 __fastcall sub_6319F0(_QWORD *a1, __int64 a2, const __m128i *a3, __int64 *a4)
{
  __int64 v5; // r12
  char v7; // r14
  __int64 result; // rax
  char v9; // dl
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // rdi
  _QWORD *v14; // rax
  _QWORD *v15; // rdi
  int v16; // edx
  _QWORD *v17; // rdi
  __int64 v18; // rax
  _QWORD *v19; // rdi
  _QWORD *v20; // rax
  char v21; // dl
  __int64 v22; // rdx
  __int64 v23; // rax
  _QWORD *v24; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v25[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a2;
  v24 = a1;
  v7 = sub_6E1B40();
  result = *((unsigned __int8 *)a1 + 8);
  if ( (_BYTE)result != 2 )
  {
    if ( (a3[2].m128i_i8[8] & 0x40) != 0 )
    {
      *a4 = 0;
      return result;
    }
    if ( (_BYTE)result )
    {
      if ( (_BYTE)result != 1 )
        sub_721090(a1);
      v9 = *(_BYTE *)(a2 + 140);
      v10 = *(_QWORD *)&dword_4D03B80;
      if ( v9 == 12 )
      {
        v11 = a2;
        do
        {
          v11 = *(_QWORD *)(v11 + 160);
          v9 = *(_BYTE *)(v11 + 140);
        }
        while ( v9 == 12 );
      }
      if ( !v9 )
        v10 = a2;
      v12 = sub_724D50(10);
      v13 = v24;
      *a4 = v12;
      *(_QWORD *)(v12 + 128) = a2;
      v14 = (_QWORD *)sub_6E1A20(v13);
      v15 = v24;
      *(_QWORD *)(*a4 + 64) = *v14;
      if ( *((_BYTE *)v15 + 8) != 2 )
      {
        v20 = (_QWORD *)sub_6E1A60(v15);
        v15 = v24;
        *(_QWORD *)(*a4 + 112) = *v20;
      }
      v16 = ~a3[2].m128i_i8[11] & 0x20;
      result = v16 | *(_BYTE *)(*a4 + 169) & 0xDFu;
      *(_BYTE *)(*a4 + 169) = v16 | *(_BYTE *)(*a4 + 169) & 0xDF;
      v17 = (_QWORD *)v15[3];
      v24 = v17;
      if ( v17 )
      {
        while ( 1 )
        {
          result = sub_6319F0(v17, v10, a3, v25);
          if ( v25[0] )
            result = sub_72A690(v25[0], *a4, 0, 0);
          v17 = (_QWORD *)*v24;
          if ( !*v24 )
            break;
          if ( *((_BYTE *)v17 + 8) == 3 )
          {
            result = sub_6BBB10(v24);
            v24 = (_QWORD *)result;
            v17 = (_QWORD *)result;
            if ( !result )
              goto LABEL_24;
          }
          else
          {
            v24 = (_QWORD *)*v24;
          }
        }
        v24 = 0;
      }
      goto LABEL_24;
    }
    if ( (unsigned int)sub_8D3DE0(a2) )
    {
      result = sub_694E20(v24, a3);
      *a4 = result;
      if ( (a3[2].m128i_i8[9] & 4) != 0 )
      {
        result = a3[1].m128i_i64[0];
        if ( result )
        {
          result = *(_QWORD *)result;
          if ( result )
          {
            v21 = *(_BYTE *)(result + 80);
            if ( v21 == 9 || v21 == 7 )
            {
              v22 = *(_QWORD *)(result + 88);
LABEL_41:
              if ( v22 )
              {
                *(_BYTE *)(result + 81) |= 1u;
                *(_BYTE *)(v22 + 88) |= 4u;
                *(_BYTE *)(v22 + 169) |= 0x10u;
              }
              goto LABEL_24;
            }
            if ( v21 == 21 )
            {
              v22 = *(_QWORD *)(*(_QWORD *)(result + 88) + 192LL);
              goto LABEL_41;
            }
          }
        }
      }
    }
    else
    {
      result = sub_631120((__int64 *)&v24, a2, a3, (__int64)a4);
    }
LABEL_24:
    if ( (a3[2].m128i_i8[8] & 0x40) == 0 )
    {
      if ( *a4 )
      {
        *(_BYTE *)(*a4 + 170) = (4 * (v7 & 1)) | *(_BYTE *)(*a4 + 170) & 0xFB;
        result = *a4;
        *(_BYTE *)(*a4 + 171) |= 4u;
      }
    }
    return result;
  }
  while ( 1 )
  {
    result = *(unsigned __int8 *)(v5 + 140);
    if ( (_BYTE)result != 12 )
      break;
    v5 = *(_QWORD *)(v5 + 160);
  }
  if ( !(_BYTE)result )
  {
    a3[2].m128i_i8[9] |= 2u;
    *a4 = 0;
    return result;
  }
  if ( (a3[2].m128i_i8[8] & 0x40) == 0 )
  {
    v18 = sub_724D50(13);
    *a4 = v18;
    v19 = v24;
    *(_QWORD *)(v18 + 128) = sub_72CBE0();
    *(_BYTE *)(*a4 + 176) |= 2u;
    if ( v19[3] )
    {
      *(_BYTE *)(*a4 + 176) |= 1u;
      result = *a4;
      *(_QWORD *)(*a4 + 184) = *(_QWORD *)(v19[3] + 8LL);
    }
    else
    {
      v23 = sub_6E1A20(v19);
      result = sub_6851C0(1797, v23);
    }
    goto LABEL_24;
  }
  return result;
}
