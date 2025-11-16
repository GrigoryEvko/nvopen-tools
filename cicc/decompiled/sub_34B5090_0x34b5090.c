// Function: sub_34B5090
// Address: 0x34b5090
//
void __fastcall sub_34B5090(_QWORD *a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 v7; // r8
  __int64 v8; // rcx
  _QWORD *v9; // rdi
  unsigned int v10; // ebx
  unsigned int *v11; // rdx
  unsigned int v12; // esi
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  __int64 v15; // [rsp+8h] [rbp-68h]
  __int64 v16; // [rsp+10h] [rbp-60h] BYREF
  int v17; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v18; // [rsp+20h] [rbp-50h]
  int *v19; // [rsp+28h] [rbp-48h]
  int *v20; // [rsp+30h] [rbp-40h]
  __int64 v21; // [rsp+38h] [rbp-38h]

  v19 = &v17;
  v20 = &v17;
  v17 = 0;
  v18 = 0;
  v21 = 0;
  sub_34B3C60((__int64)a1, a2, &v16);
  sub_34B48B0((__int64)a1, a2, a3, (__int64)&v16);
  sub_34B4E10(a1, a2, a3);
  v7 = a1[4];
  v8 = a1[15];
  if ( *(_DWORD *)(v7 + 16) != 1 )
  {
    v9 = (_QWORD *)a1[15];
    v10 = 1;
    while ( 1 )
    {
      if ( *(_DWORD *)(v9[13] + 4LL * v10) == -1 || *(_DWORD *)(v9[16] + 4LL * v10) != -1 )
      {
        v11 = (unsigned int *)(*(_QWORD *)(v8 + 128) + 4LL * v10);
        if ( *v11 < a4 && *v11 >= a3 )
        {
          *v11 = a3;
          v7 = a1[4];
        }
        if ( *(_DWORD *)(v7 + 16) == ++v10 )
          break;
      }
      else
      {
        v12 = v10;
        v15 = v8;
        ++v10;
        sub_34B3410(v9, v12, 0);
        v7 = a1[4];
        v8 = v15;
        if ( *(_DWORD *)(v7 + 16) == v10 )
          break;
      }
      v9 = (_QWORD *)a1[15];
    }
  }
  v13 = v18;
  while ( v13 )
  {
    sub_34B4200((__int64)&v16, *(_QWORD *)(v13 + 24));
    v14 = v13;
    v13 = *(_QWORD *)(v13 + 16);
    j_j___libc_free_0(v14);
  }
}
