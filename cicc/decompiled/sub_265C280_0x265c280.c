// Function: sub_265C280
// Address: 0x265c280
//
void __fastcall sub_265C280(__int64 a1, __int64 a2)
{
  unsigned __int64 *v2; // r15
  unsigned __int64 v3; // rdx
  _QWORD *v4; // r12
  unsigned __int64 v5; // r9
  int v6; // r14d
  __int64 v7; // rcx
  unsigned __int64 *v8; // rax
  _QWORD *v9; // rcx
  unsigned __int64 *v10; // rax
  int v11; // eax
  unsigned __int64 *v12; // rax
  __int64 v13; // [rsp-50h] [rbp-50h]
  unsigned __int64 v14; // [rsp-50h] [rbp-50h]
  int v15; // [rsp-50h] [rbp-50h]
  unsigned __int64 v16; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 != a2 )
  {
    v2 = (unsigned __int64 *)(a2 + 16);
    v3 = *(unsigned int *)(a1 + 8);
    v4 = *(_QWORD **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v5 = *(unsigned int *)(a2 + 8);
      v6 = *(_DWORD *)(a2 + 8);
      if ( v5 <= v3 )
      {
        v10 = *(unsigned __int64 **)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v12 = sub_265BF70(v2, (__int64)&v2[v5], *(unsigned __int64 **)a1);
          v3 = *(unsigned int *)(a1 + 8);
          v4 = v12;
          v10 = *(unsigned __int64 **)a1;
        }
        sub_2649CB0((__int64)v4, (__int64)&v10[v3]);
      }
      else
      {
        v7 = v3;
        if ( v5 > *(unsigned int *)(a1 + 12) )
        {
          v14 = *(unsigned int *)(a2 + 8);
          sub_2649CB0(*(_QWORD *)a1, (__int64)&v4[v7]);
          *(_DWORD *)(a1 + 8) = 0;
          v4 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, v14, 8u, &v16, v14);
          sub_264AB90(a1, v4);
          v11 = v16;
          if ( a1 + 16 != *(_QWORD *)a1 )
          {
            v15 = v16;
            _libc_free(*(_QWORD *)a1);
            v11 = v15;
          }
          *(_DWORD *)(a1 + 12) = v11;
          *(_QWORD *)a1 = v4;
          v2 = *(unsigned __int64 **)a2;
          v5 = *(unsigned int *)(a2 + 8);
          v8 = *(unsigned __int64 **)a2;
        }
        else
        {
          v8 = (unsigned __int64 *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v13 = 8 * v3;
            sub_265BF70(v2, (__int64)&v2[v7], *(unsigned __int64 **)a1);
            v2 = *(unsigned __int64 **)a2;
            v5 = *(unsigned int *)(a2 + 8);
            v4 = (_QWORD *)(v13 + *(_QWORD *)a1);
            v8 = (unsigned __int64 *)(*(_QWORD *)a2 + v13);
          }
        }
        v9 = (_QWORD *)((char *)v4 + (char *)&v2[v5] - (char *)v8);
        if ( &v2[v5] != v8 )
        {
          do
          {
            if ( v4 )
            {
              *v4 = *v8;
              *v8 = 0;
            }
            ++v4;
            ++v8;
          }
          while ( v4 != v9 );
        }
      }
      *(_DWORD *)(a1 + 8) = v6;
      sub_2649CB0(*(_QWORD *)a2, *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8));
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      sub_2649CB0(*(_QWORD *)a1, (__int64)&v4[v3]);
      if ( *(_QWORD *)a1 != a1 + 16 )
        _libc_free(*(_QWORD *)a1);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v2;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
