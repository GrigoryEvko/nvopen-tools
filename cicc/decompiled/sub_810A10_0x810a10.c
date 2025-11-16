// Function: sub_810A10
// Address: 0x810a10
//
void __fastcall sub_810A10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v6; // r12
  size_t v7; // rax
  char *v8; // rax
  char *v9; // rax
  _QWORD v10[4]; // [rsp-78h] [rbp-78h] BYREF
  char v11; // [rsp-58h] [rbp-58h]
  __int64 v12; // [rsp-50h] [rbp-50h]
  __int64 v13; // [rsp-48h] [rbp-48h]
  int v14; // [rsp-40h] [rbp-40h]
  char v15; // [rsp-3Ch] [rbp-3Ch]
  __int64 v16; // [rsp-38h] [rbp-38h]

  if ( *(_QWORD *)(a1 + 8) )
  {
    v10[3] = 0;
    v11 = 0;
    v12 = 0;
    v13 = 0;
    v14 = 0;
    v15 = 0;
    v16 = 0;
    sub_809110(a1, a2, a3, a4, a5, a6, 0, 0, 0);
    sub_823800(qword_4F18BE0);
    v10[0] += 6LL;
    sub_8238B0(qword_4F18BE0, "__SO__", 6);
    sub_810650(a1, 1, v10);
    v6 = sub_80B290(0, 0, (__int64)v10);
    v7 = strlen((const char *)v6);
    v8 = (char *)sub_7E1510(v7 + 1);
    v9 = strcpy(v8, (const char *)v6);
    *(_BYTE *)(a2 + 89) |= 8u;
    *(_QWORD *)(a2 + 8) = v9;
  }
}
