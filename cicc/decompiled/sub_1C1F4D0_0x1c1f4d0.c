// Function: sub_1C1F4D0
// Address: 0x1c1f4d0
//
__int64 __fastcall sub_1C1F4D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rbx
  char i; // al
  char *v7; // rdx
  unsigned __int64 v8; // rax
  char *v9; // r14
  unsigned int v10; // eax
  unsigned __int8 v11; // al
  char *v13; // rax
  __int64 v14; // [rsp+28h] [rbp-58h]
  char v15; // [rsp+3Bh] [rbp-45h] BYREF
  int v16; // [rsp+3Ch] [rbp-44h] BYREF
  __int64 v17; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v18[7]; // [rsp+48h] [rbp-38h] BYREF

  LODWORD(v2) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v2 = (__int64)(*(_QWORD *)(a2 + 8) - *(_QWORD *)a2) >> 3;
  if ( (_DWORD)v2 )
  {
    v3 = (unsigned int)(v2 - 1);
    v4 = 1;
    v5 = 0;
    v14 = v3 + 2;
    for ( i = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(a1, 0, &v17);
          ;
          i = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(a1, (unsigned int)v5, &v17) )
    {
      if ( i )
      {
        v7 = *(char **)a2;
        v8 = (__int64)(*(_QWORD *)(a2 + 8) - *(_QWORD *)a2) >> 3;
        if ( v8 <= v5 )
        {
          if ( v8 < v4 )
          {
            sub_1C1F320((char **)a2, v4 - v8);
            v7 = *(char **)a2;
          }
          else if ( v8 > v4 )
          {
            v13 = &v7[8 * v4];
            if ( *(char **)(a2 + 8) != v13 )
              *(_QWORD *)(a2 + 8) = v13;
          }
        }
        v9 = &v7[8 * v5];
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
        v16 = *(_DWORD *)v9 & 0xFFFFFF;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "Reg",
               1,
               0,
               &v15,
               v18) )
        {
          sub_1C14710(a1, (unsigned int *)&v16);
          (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v18[0]);
        }
        v10 = *(_DWORD *)v9 & 0xFF000000;
        *(_DWORD *)v9 = v10 | v16 & 0xFFFFFF;
        v16 = HIBYTE(v10) & 0x1F;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "RegClass",
               1,
               0,
               &v15,
               v18) )
        {
          sub_1C14710(a1, (unsigned int *)&v16);
          (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v18[0]);
        }
        v11 = v16 & 0x1F | v9[3] & 0xE0;
        v9[3] = v11;
        v16 = v11 >> 5;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "RegComp",
               1,
               0,
               &v15,
               v18) )
        {
          sub_1C14710(a1, (unsigned int *)&v16);
          (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v18[0]);
        }
        v9[3] = (32 * v16) | v9[3] & 0x1F;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, int *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "LogAlign",
               1,
               0,
               &v16,
               v18) )
        {
          sub_1C14710(a1, (unsigned int *)v9 + 1);
          (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v18[0]);
        }
        ++v5;
        ++v4;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v17);
        if ( v14 == v4 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
      else
      {
        ++v5;
        if ( v14 == ++v4 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
    }
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
}
