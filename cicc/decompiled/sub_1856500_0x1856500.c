// Function: sub_1856500
// Address: 0x1856500
//
const char *sub_1856500()
{
  unsigned __int64 v0; // rdx
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // rcx
  const char *result; // rax
  const char *v4; // rdx
  const char *v5; // [rsp+0h] [rbp-10h] BYREF
  unsigned __int64 v6; // [rsp+8h] [rbp-8h]

  v5 = "llvm70::StringRef llvm70::getTypeName() [with DesiredTypeName = llvm70::InnerAnalysisManagerProxy<llvm70::Analysi"
       "sManager<llvm70::Function>, llvm70::Module>]";
  v6 = 157;
  v0 = sub_16D20C0((__int64 *)&v5, "DesiredTypeName = ", 0x12u, 0);
  v1 = v6;
  if ( v0 > v6 )
    return &v5[v6];
  v2 = v6 - v0;
  if ( v6 - v0 == -1 )
  {
    result = &v5[v0 + 18];
LABEL_4:
    if ( *(_QWORD *)result == 0x3A3A534E4D564C4CLL )
      result += 8;
    return result;
  }
  v4 = &v5[v0];
  v6 = v2;
  result = &v5[v1];
  v5 = v4;
  if ( v2 > 0x11 )
  {
    result = v4 + 18;
    if ( v2 - 18 >= v2 - 19 && v2 - 19 > 7 )
      goto LABEL_4;
  }
  return result;
}
