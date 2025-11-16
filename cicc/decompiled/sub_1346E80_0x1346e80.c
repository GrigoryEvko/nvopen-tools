// Function: sub_1346E80
// Address: 0x1346e80
//
__int64 __fastcall sub_1346E80(unsigned int a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v9; // rdi
  _BYTE *v10; // r11
  __int64 *v11; // r15
  __int64 v12; // r10
  __int64 *v13; // r9
  __int64 v14; // rsi
  __int64 *v15; // rdx
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 *v18; // [rsp-80h] [rbp-80h]
  _BYTE *v19; // [rsp-78h] [rbp-78h]
  __int64 v20; // [rsp-70h] [rbp-70h]
  _QWORD v21[4]; // [rsp-68h] [rbp-68h] BYREF
  char v22; // [rsp-48h] [rbp-48h]
  __int64 v23; // [rsp-40h] [rbp-40h] BYREF

  result = (unsigned int)dword_4F96DA0;
  if ( !dword_4F96DA0 )
    return result;
  result = (__int64)byte_4F96B58;
  if ( !byte_4F96B58[0] )
    goto LABEL_16;
  result = -2664;
  v9 = __readfsqword(0) - 2664;
  v10 = (_BYTE *)(v9 + 216);
  if ( !__readfsbyte(0xFFFFF8C8) )
    goto LABEL_4;
  result = sub_1313D30(v9, 0);
  if ( result )
    v10 = (_BYTE *)(result + 216);
  else
LABEL_16:
    v10 = &unk_4C6F2C9;
LABEL_4:
  if ( !*v10 )
  {
    *v10 = 1;
    v11 = (__int64 *)&unk_4F96CE0;
    v12 = 0;
    v13 = &v23;
    do
    {
      v14 = *v11;
      if ( (*v11 & 1) == 0 )
      {
        v15 = v11 + 1;
        v16 = v21;
        do
        {
          v17 = *v15;
          ++v16;
          ++v15;
          *(v16 - 1) = v17;
        }
        while ( v13 != v16 );
        result = *v11;
        if ( v14 == *v11 )
        {
          result = v21[0];
          if ( v22 )
          {
            if ( v21[0] )
            {
              v18 = v13;
              v19 = v10;
              v20 = v12;
              result = ((__int64 (__fastcall *)(_QWORD, _QWORD, __int64, __int64, __int64))v21[0])(
                         v21[3],
                         a1,
                         a2,
                         a3,
                         a4);
              v13 = v18;
              v10 = v19;
              v12 = v20;
            }
          }
        }
      }
      v12 += 6;
      v11 += 6;
    }
    while ( v12 != 24 );
    *v10 = 0;
  }
  return result;
}
