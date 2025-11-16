// Function: sub_3504CA0
// Address: 0x3504ca0
//
unsigned __int64 __fastcall sub_3504CA0(__int64 a1, _BYTE *a2)
{
  _QWORD *v2; // rax
  unsigned __int64 result; // rax
  unsigned __int8 v4; // dl
  __int64 v5; // rax
  __int64 v6; // r9
  __int64 v7; // rdx
  unsigned __int64 v8; // [rsp+0h] [rbp-70h]
  _BYTE *v9; // [rsp+8h] [rbp-68h] BYREF
  char v10; // [rsp+17h] [rbp-59h] BYREF
  __int64 v11; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 *v12; // [rsp+20h] [rbp-50h] BYREF
  __int64 v13; // [rsp+28h] [rbp-48h] BYREF
  _QWORD v14[8]; // [rsp+30h] [rbp-40h] BYREF

  v9 = sub_AF3520(a2);
  v2 = sub_32239E0((_QWORD *)(a1 + 120), (__int64 *)&v9);
  if ( v2 )
    return (unsigned __int64)(v2 + 2);
  v11 = 0;
  if ( (unsigned int)(unsigned __int8)*v9 - 19 <= 1 )
  {
    v4 = *(v9 - 16);
    if ( (v4 & 2) != 0 )
      v5 = *((_QWORD *)v9 - 4);
    else
      v5 = (__int64)&v9[-8 * ((v4 >> 2) & 0xF) - 16];
    v11 = sub_3504CA0(a1, *(_QWORD *)(v5 + 8));
  }
  v10 = 1;
  v14[0] = &v10;
  v14[1] = &v13;
  v13 = 0;
  v14[2] = &v9;
  v14[3] = &v11;
  v12 = (unsigned __int64 *)&v9;
  result = sub_35045A0((unsigned __int64 *)(a1 + 120), &v12, (__int64)v14) + 16;
  if ( *v9 == 18 )
  {
    v7 = *(unsigned int *)(a1 + 184);
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 188) )
    {
      v8 = result;
      sub_C8D5F0(a1 + 176, (const void *)(a1 + 192), v7 + 1, 8u, v7 + 1, v6);
      v7 = *(unsigned int *)(a1 + 184);
      result = v8;
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 176) + 8 * v7) = result;
    ++*(_DWORD *)(a1 + 184);
  }
  return result;
}
