// Function: sub_80B6E0
// Address: 0x80b6e0
//
void __fastcall sub_80B6E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rdi
  char *v5; // r14
  size_t v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int8 *v11; // rax
  unsigned __int8 *v12; // rdx
  __int64 v13; // rax
  _QWORD v14[4]; // [rsp-78h] [rbp-78h] BYREF
  char v15; // [rsp-58h] [rbp-58h]
  __int64 v16; // [rsp-50h] [rbp-50h]
  __int64 v17; // [rsp-48h] [rbp-48h]
  int v18; // [rsp-40h] [rbp-40h]
  char v19; // [rsp-3Ch] [rbp-3Ch]
  __int64 v20; // [rsp-38h] [rbp-38h]

  if ( a1 )
  {
    v2 = a1;
    do
    {
      if ( !(unsigned int)sub_736DD0(v2) )
      {
        if ( (unsigned __int8)(*(_BYTE *)(v2 + 140) - 9) <= 2u )
        {
          v3 = *(_QWORD *)(v2 + 168);
          v4 = *(_QWORD *)(v3 + 152);
          if ( v4 && (*(_BYTE *)(v4 + 29) & 0x20) == 0 )
            sub_80B480((_QWORD *)v4, a2);
          sub_80B6E0(*(_QWORD *)(v3 + 216));
        }
        if ( (*(_BYTE *)(v2 + 89) & 0x20) != 0 )
        {
          v5 = *(char **)(v2 + 8);
          v6 = strlen(v5) + 1;
          v14[3] = 0;
          v15 = 0;
          v16 = 0;
          v17 = 0;
          v18 = 0;
          v19 = 0;
          v20 = 0;
          sub_809110(v5, a2, v7, v8, v9, v10, 0, 0, 0);
          sub_823800(qword_4F18BE0);
          v14[0] = v6;
          v11 = sub_809930((unsigned __int8 *)v5, v2, (__int64)v14);
          a2 = qword_4F18BE8;
          v12 = v11;
          v13 = *(_QWORD *)qword_4F18BF0;
          qword_4F18BE8 = qword_4F18BF0;
          *(_QWORD *)qword_4F18BF0 = a2;
          qword_4F18BF0 = v13;
          if ( v13 )
            v13 = *(_QWORD *)(v13 + 8);
          qword_4F18BE0 = v13;
          *(_QWORD *)(v2 + 8) = v12;
          *(_BYTE *)(v2 + 89) &= ~0x20u;
        }
      }
      v2 = *(_QWORD *)(v2 + 112);
    }
    while ( v2 );
  }
}
