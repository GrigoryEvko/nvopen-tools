// Function: sub_1346FC0
// Address: 0x1346fc0
//
__int64 __fastcall sub_1346FC0(unsigned int a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v7; // rdi
  _BYTE *v8; // r9
  __int64 *v9; // r14
  __int64 i; // r8
  __int64 v11; // rsi
  __int64 *v12; // rdx
  __int64 *v13; // rax
  __int64 v14; // rcx
  _BYTE *v15; // [rsp-78h] [rbp-78h]
  __int64 v16; // [rsp-70h] [rbp-70h]
  __int64 v17; // [rsp-68h] [rbp-68h] BYREF
  __int64 (__fastcall *v18)(__int64, _QWORD, __int64, __int64); // [rsp-60h] [rbp-60h]
  __int64 v19; // [rsp-50h] [rbp-50h]
  char v20; // [rsp-48h] [rbp-48h]
  __int64 v21; // [rsp-40h] [rbp-40h] BYREF

  result = (unsigned int)dword_4F96DA0;
  if ( !dword_4F96DA0 )
    return result;
  result = (__int64)byte_4F96B58;
  if ( !byte_4F96B58[0] )
    goto LABEL_16;
  result = -2664;
  v7 = __readfsqword(0) - 2664;
  v8 = (_BYTE *)(v7 + 216);
  if ( !__readfsbyte(0xFFFFF8C8) )
    goto LABEL_4;
  result = sub_1313D30(v7, 0);
  if ( result )
    v8 = (_BYTE *)(result + 216);
  else
LABEL_16:
    v8 = &unk_4C6F2C9;
LABEL_4:
  if ( !*v8 )
  {
    *v8 = 1;
    v9 = (__int64 *)&unk_4F96CE0;
    for ( i = 0; i != 24; i += 6 )
    {
      v11 = *v9;
      if ( (*v9 & 1) == 0 )
      {
        v12 = v9 + 1;
        v13 = &v17;
        do
        {
          v14 = *v12;
          ++v13;
          ++v12;
          *(v13 - 1) = v14;
        }
        while ( &v21 != v13 );
        result = *v9;
        if ( v11 == *v9 )
        {
          result = (__int64)v18;
          if ( v20 )
          {
            if ( v18 )
            {
              v15 = v8;
              v16 = i;
              result = v18(v19, a1, a2, a3);
              v8 = v15;
              i = v16;
            }
          }
        }
      }
      v9 += 6;
    }
    *v8 = 0;
  }
  return result;
}
