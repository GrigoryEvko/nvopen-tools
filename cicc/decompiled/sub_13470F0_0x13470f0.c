// Function: sub_13470F0
// Address: 0x13470f0
//
unsigned __int64 __fastcall sub_13470F0(unsigned int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 result; // rax
  __int64 v10; // rdi
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r15
  unsigned int v14; // r10d
  __int64 v15; // r13
  __int64 *v16; // rbx
  __int64 v17; // rdi
  __int64 *v18; // rdx
  __int64 *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // [rsp-88h] [rbp-88h]
  __int64 v23; // [rsp-80h] [rbp-80h]
  unsigned int v24; // [rsp-74h] [rbp-74h]
  _BYTE *v25; // [rsp-70h] [rbp-70h]
  __int64 v26; // [rsp-68h] [rbp-68h] BYREF
  void (__fastcall *v27)(__int64, _QWORD, __int64, __int64); // [rsp-58h] [rbp-58h]
  __int64 v28; // [rsp-50h] [rbp-50h]
  char v29; // [rsp-48h] [rbp-48h]
  __int64 v30; // [rsp-40h] [rbp-40h] BYREF

  result = (unsigned int)dword_4F96DA0;
  if ( !dword_4F96DA0 )
    return result;
  if ( !byte_4F96B58[0] )
    goto LABEL_16;
  v10 = __readfsqword(0) - 2664;
  v25 = (_BYTE *)(v10 + 216);
  if ( !__readfsbyte(0xFFFFF8C8) )
    goto LABEL_4;
  v21 = sub_1313D30(v10, 0);
  if ( v21 )
    v25 = (_BYTE *)(v21 + 216);
  else
LABEL_16:
    v25 = &unk_4C6F2C9;
LABEL_4:
  result = (unsigned __int64)v25;
  if ( !*v25 )
  {
    *v25 = 1;
    v11 = a4;
    v12 = a5;
    v13 = a3;
    v14 = a1;
    v15 = 0;
    v16 = (__int64 *)&unk_4F96CE0;
    do
    {
      v17 = *v16;
      if ( (*v16 & 1) == 0 )
      {
        v18 = v16 + 1;
        v19 = &v26;
        do
        {
          v20 = *v18;
          ++v19;
          ++v18;
          *(v19 - 1) = v20;
        }
        while ( &v30 != v19 );
        if ( v17 == *v16 && v29 )
        {
          if ( v27 )
          {
            v22 = v12;
            v23 = v11;
            v24 = v14;
            v27(v28, v14, a2, v13);
            v12 = v22;
            v11 = v23;
            v14 = v24;
          }
        }
      }
      v15 += 6;
      v16 += 6;
    }
    while ( v15 != 24 );
    result = (unsigned __int64)v25;
    *v25 = 0;
  }
  return result;
}
