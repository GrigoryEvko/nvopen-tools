// Function: sub_22BECA0
// Address: 0x22beca0
//
__int64 __fastcall sub_22BECA0(unsigned __int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  unsigned int v5; // edx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 result; // rax
  int v9; // ecx
  unsigned int v10; // esi
  int v11; // eax
  unsigned __int64 *v12; // r15
  int v13; // eax
  bool v14; // zf
  int v15; // r10d
  unsigned __int64 *v16; // [rsp+0h] [rbp-A0h] BYREF
  unsigned __int64 *v17; // [rsp+8h] [rbp-98h] BYREF
  __int64 (__fastcall **v18)(); // [rsp+10h] [rbp-90h] BYREF
  _QWORD v19[3]; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int64 v20; // [rsp+30h] [rbp-70h]
  void *v21; // [rsp+40h] [rbp-60h]
  _QWORD v22[11]; // [rsp+48h] [rbp-58h] BYREF

  v3 = *(unsigned int *)(a1 + 56);
  if ( !(_DWORD)v3 )
    goto LABEL_7;
  v4 = *(_QWORD *)(a1 + 40);
  v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = v4 + 40LL * v5;
  v7 = *(_QWORD *)(v6 + 24);
  if ( v7 != a2 )
  {
    v9 = 1;
    while ( v7 != -4096 )
    {
      v15 = v9 + 1;
      v5 = (v3 - 1) & (v9 + v5);
      v6 = v4 + 40LL * v5;
      v7 = *(_QWORD *)(v6 + 24);
      if ( v7 == a2 )
        goto LABEL_3;
      v9 = v15;
    }
LABEL_7:
    v19[2] = a2;
    v19[0] = 2;
    v19[1] = 0;
    if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
      sub_BD73F0((__int64)v19);
    v20 = a1;
    v18 = off_4A09D90;
    if ( (unsigned __int8)sub_22BDFD0(a1 + 32, (__int64)&v18, &v16) )
      goto LABEL_11;
    v10 = *(_DWORD *)(a1 + 56);
    v11 = *(_DWORD *)(a1 + 48);
    v12 = v16;
    ++*(_QWORD *)(a1 + 32);
    v13 = v11 + 1;
    v17 = v12;
    if ( 4 * v13 >= 3 * v10 )
    {
      v10 *= 2;
    }
    else if ( v10 - *(_DWORD *)(a1 + 52) - v13 > v10 >> 3 )
    {
LABEL_14:
      *(_DWORD *)(a1 + 48) = v13;
      v22[2] = -4096;
      v22[3] = 0;
      v14 = v12[3] == -4096;
      v22[0] = 2;
      v22[1] = 0;
      if ( !v14 )
        --*(_DWORD *)(a1 + 52);
      v21 = &unk_49DB368;
      sub_D68D70(v22);
      sub_22BDBB0(v12 + 1, v19);
      v12[4] = v20;
LABEL_11:
      v18 = (__int64 (__fastcall **)())&unk_49DB368;
      return sub_D68D70(v19);
    }
    sub_22BE150(a1 + 32, v10);
    sub_22BDFD0(a1 + 32, (__int64)&v18, &v17);
    v12 = v17;
    v13 = *(_DWORD *)(a1 + 48) + 1;
    goto LABEL_14;
  }
LABEL_3:
  result = v4 + 40 * v3;
  if ( v6 == result )
    goto LABEL_7;
  return result;
}
