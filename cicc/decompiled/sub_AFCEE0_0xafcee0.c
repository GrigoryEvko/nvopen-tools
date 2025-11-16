// Function: sub_AFCEE0
// Address: 0xafcee0
//
__int64 __fastcall sub_AFCEE0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r14d
  __int64 v6; // rbx
  __int64 v7; // r15
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  int v11; // r14d
  int v12; // eax
  __int64 v13; // rsi
  int v14; // r8d
  _QWORD *v15; // rdi
  unsigned int v16; // eax
  _QWORD *v17; // rcx
  __int64 v18; // rdx
  int v19; // [rsp+0h] [rbp-60h] BYREF
  __int64 v20; // [rsp+8h] [rbp-58h] BYREF
  __int64 v21; // [rsp+10h] [rbp-50h] BYREF
  int v22; // [rsp+18h] [rbp-48h] BYREF
  int v23[17]; // [rsp+1Ch] [rbp-44h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v19 = (unsigned __int16)sub_AF18C0(*a2);
    v9 = *(_BYTE *)(v6 - 16);
    if ( (v9 & 2) != 0 )
      v10 = *(_QWORD *)(v6 - 32);
    else
      v10 = v6 - 16 - 8LL * ((v9 >> 2) & 0xF);
    v11 = v4 - 1;
    v20 = *(_QWORD *)(v10 + 16);
    v21 = *(_QWORD *)(v6 + 24);
    v22 = sub_AF18D0(v6);
    v23[0] = *(_DWORD *)(v6 + 44);
    v23[1] = *(_DWORD *)(v6 + 40);
    v23[2] = *(_DWORD *)(v6 + 20);
    v12 = sub_AF9B00(&v19, &v20, &v21, &v22, v23);
    v13 = *a2;
    v14 = 1;
    v15 = 0;
    v16 = v11 & v12;
    v17 = (_QWORD *)(v7 + 8LL * v16);
    v18 = *v17;
    if ( *a2 == *v17 )
    {
LABEL_13:
      *a3 = v17;
      return 1;
    }
    else
    {
      while ( v18 != -4096 )
      {
        if ( v18 != -8192 || v15 )
          v17 = v15;
        v16 = v11 & (v14 + v16);
        v18 = *(_QWORD *)(v7 + 8LL * v16);
        if ( v18 == v13 )
        {
          v17 = (_QWORD *)(v7 + 8LL * v16);
          goto LABEL_13;
        }
        ++v14;
        v15 = v17;
        v17 = (_QWORD *)(v7 + 8LL * v16);
      }
      if ( !v15 )
        v15 = v17;
      *a3 = v15;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
