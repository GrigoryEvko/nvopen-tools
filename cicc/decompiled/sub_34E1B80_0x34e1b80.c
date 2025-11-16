// Function: sub_34E1B80
// Address: 0x34e1b80
//
__int64 __fastcall sub_34E1B80(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, size_t a5, unsigned __int8 a6)
{
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 i; // r12
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // [rsp+8h] [rbp-78h]
  __int64 v13; // [rsp+10h] [rbp-70h]
  __int64 v19; // [rsp+40h] [rbp-40h] BYREF
  __int64 v20[7]; // [rsp+48h] [rbp-38h] BYREF

  result = a6;
  v7 = *(_QWORD *)(a1 + 152);
  v13 = *(_QWORD *)(v7 + 328);
  v12 = v7 + 320;
  if ( v13 != v7 + 320 )
  {
    do
    {
      v8 = *(_QWORD *)(v13 + 56);
      for ( i = v13 + 48; i != v8; v8 = *(_QWORD *)(v8 + 8) )
      {
        while ( 1 )
        {
          if ( (unsigned __int16)(*(_WORD *)(v8 + 68) - 14) <= 2u )
          {
            v10 = sub_2E89170(v8);
            if ( v10 )
            {
              v11 = *(_QWORD *)(v8 + 56);
              v19 = v11;
              if ( v11 )
              {
                sub_B96E90((__int64)&v19, v11, 1);
                v20[0] = v19;
                if ( v19 )
                  sub_B96E90((__int64)v20, v19, 1);
              }
              else
              {
                v20[0] = 0;
              }
              sub_3142D90(a1, v10, (__int64)v20, a2, a3, a6, a4, a5);
              if ( v20[0] )
                sub_B91220((__int64)v20, v20[0]);
              if ( v19 )
                sub_B91220((__int64)&v19, v19);
            }
          }
          if ( (*(_BYTE *)v8 & 4) == 0 )
            break;
          v8 = *(_QWORD *)(v8 + 8);
          if ( i == v8 )
            goto LABEL_16;
        }
        while ( (*(_BYTE *)(v8 + 44) & 8) != 0 )
          v8 = *(_QWORD *)(v8 + 8);
      }
LABEL_16:
      result = *(_QWORD *)(v13 + 8);
      v13 = result;
    }
    while ( v12 != result );
  }
  return result;
}
